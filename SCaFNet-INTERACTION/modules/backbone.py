from typing import Tuple

import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import dense_to_sparse
import torch.nn.functional as F
import math
from layers import GraphAttention
from layers import TwoLayerMLP
from utils import compute_angles_lengths_2D
from utils import init_weights
from utils import wrap_angle
from utils import drop_edge_between_samples
from utils import transform_point_to_local_coordinate
from utils import transform_point_to_global_coordinate
from utils import transform_traj_to_global_coordinate
from utils import transform_traj_to_local_coordinate


class SceneMLPBlock(nn.Module):
    """
    两层残差 MLP：LN -> Linear -> GELU -> Dropout? -> Linear
    第二层零初始化：一开始输出≈0，当作残差支路用
    """

    def __init__(self, hidden_dim: int, expansion: int = 4, dropout: float = 0.0):
        super().__init__()
        inner = hidden_dim * expansion

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, inner)
        self.fc2 = nn.Linear(inner, hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

        # 第一层：正常初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # 第二层：零初始化，刚开始基本不改变输入
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return h  # 外面还是用 x + scene_mlp(x)


class GatedMLP1D(nn.Module):
    """
    小 MLP gate：LN -> Linear -> GELU -> Linear
    输出是 gate 的 logits（形状 [...,1]），
    外面再套一次 torch.sigmoid(...) 得到真正的 gate。
    """

    def __init__(self, hidden_dim: int, gate_hidden: int = None):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(16, hidden_dim // 4)

        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, gate_hidden)
        self.fc2 = nn.Linear(gate_hidden, 1)
        self.act = nn.GELU()

        # 第一层：正常初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)

        # 第二层：小增益初始化，让初始 gate 接近 0.5 区域，梯度更平滑
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        # x: [..., hidden_dim]
        h = self.norm(x)
        h = self.act(self.fc1(h))
        logits = self.fc2(h)  # [..., 1]，不做 sigmoid
        return logits


class Backbone(nn.Module):

    def __init__(self,
                 hidden_dim: int,
                 num_historical_steps: int,
                 num_future_steps: int,
                 duration: int,
                 a2a_radius: float,
                 l2a_radius: float,
                 num_attn_layers: int,
                 num_modes: int,
                 num_heads: int,
                 dropout: float) -> None:
        super(Backbone, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_historical_steps = num_historical_steps
        self.num_future_steps = num_future_steps
        self.duration = duration
        self.a2a_radius = a2a_radius
        self.l2a_radius = l2a_radius
        self.num_attn_layers = num_attn_layers
        self.num_modes = num_modes
        self.num_heads = num_heads
        self.dropout = dropout

        self.mode_tokens = nn.Embedding(num_modes, hidden_dim)  # [K,D]

        self.a_emb_layer = TwoLayerMLP(input_dim=5, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.t2m_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.m2m_h_emb_layer = TwoLayerMLP(input_dim=4, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_a_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)
        self.m2m_s_emb_layer = TwoLayerMLP(input_dim=3, hidden_dim=hidden_dim, output_dim=hidden_dim)

        self.l2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=False)
        self.t2m_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=False)

        self.m2m_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.m2m_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=False,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_propose = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                        output_dim=self.num_future_steps * 2)

        self.proposal_to_anchor = TwoLayerMLP(input_dim=self.num_future_steps * 2, hidden_dim=hidden_dim,
                                              output_dim=hidden_dim)

        self.l2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=False)
        self.t2n_attn_layer = GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout,
                                             has_edge_attr=True, if_self_attention=False)

        self.n2n_h_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_a_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])
        self.n2n_s_attn_layers = nn.ModuleList([GraphAttention(hidden_dim=hidden_dim, num_heads=num_heads,
                                                               dropout=dropout, has_edge_attr=True,
                                                               if_self_attention=True) for _ in range(num_attn_layers)])

        self.traj_refine = TwoLayerMLP(input_dim=hidden_dim, hidden_dim=hidden_dim,
                                       output_dim=self.num_future_steps * 2)

        # 1) Trajectory Memory：把标量初始化改为“更平滑”的起点（例如 0.9）
        #
        # ema_alpha_init = 0.90  # ← 想更灵敏改小点，比如 0.8；想更平滑改大点，比如 0.95
        # self.tm_alpha = nn.Parameter(torch.full((hidden_dim,),
        #                                         torch.logit(torch.tensor(ema_alpha_init))))
        #
        # # 2) Scene Context Memory：可调宽度的残差适配器（保持调用 self.scene_mlp(l_embs) 不变）
        # #    expand>1 放大容量；=1 大约等同“隐藏层=hidden_dim”；<1 更轻量（比如 0.5）
        # self.scene_mlp = SceneMLPScaled(hidden_dim, expand=1.0, p_drop=0.1)
        #
        # # 3) Gated Fusion：把 gate 从一层线性放大为两层 MLP，但接口不变
        # #    expand 典型取 0.5~1.0；更大=更强的门，但也更容易过拟合
        # self.fuse_gate_m = GateMLP(hidden_dim, expand=0.5)
        # self.fuse_gate_n = GateMLP(hidden_dim, expand=0.5)

        # # === Trajectory Memory：逐通道 EMA 参数（稳定且省显存）===
        # self.tm_alpha = nn.Parameter(torch.zeros(hidden_dim))  # sigmoid 后在 (0,1)
        #
        # # === Scene Context Memory：极简线性残差（零初始化，起步等价于恒等映射）===
        # self.scene_mlp = nn.Linear(hidden_dim, hidden_dim)
        # nn.init.zeros_(self.scene_mlp.weight)
        # nn.init.zeros_(self.scene_mlp.bias)
        #
        # # === Gated Fusion（差分门控，省激活）===
        # self.fuse_gate_m = nn.Linear(hidden_dim, 1)
        # self.fuse_gate_n = nn.Linear(hidden_dim, 1)

        ema_init = 0.9
        logit = math.log(ema_init / (1 - ema_init))
        # alpha 控制“记忆保留”，beta 可作为“额外调制”，forward 里你可以只用 alpha 也行
        self.tm_alpha = nn.Parameter(torch.full((hidden_dim,), logit))

        self.scene_mlp = SceneMLPBlock(hidden_dim, expansion=4, dropout=0.1)
        self.fuse_gate_m = GatedMLP1D(hidden_dim, gate_hidden=None)
        self.fuse_gate_n = GatedMLP1D(hidden_dim, gate_hidden=None)

        self.apply(init_weights)

    def forward(self, data: Batch, l_embs: torch.Tensor) -> torch.Tensor:
        # initialization
        a_velocity_length = data['agent']['velocity_length']  # [(N1,...,Nb),H]
        a_velocity_theta = data['agent']['velocity_theta']  # [(N1,...,Nb),H]
        a_length = data['agent']['length'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,
                                                                           -1)  # [(N1,...,Nb),H]
        a_width = data['agent']['width'].unsqueeze(-1).repeat_interleave(self.num_historical_steps,
                                                                         -1)  # [(N1,...,Nb),H]
        a_type = data['agent']['type'].unsqueeze(-1).repeat_interleave(self.num_historical_steps, -1)  # [(N1,...,Nb),H]
        a_input = torch.stack([a_velocity_length, a_velocity_theta, a_length, a_width, a_type], dim=-1)
        a_embs = self.a_emb_layer(input=a_input)  # [(N1,...,Nb),H,D]

        # --- Trajectory Memory（逐通道 EMA，常数显存、无 NaN）---
        vis = data['agent']['visible_mask'][:, :self.num_historical_steps].bool()  # [N,H]
        a_masked = a_embs * vis.unsqueeze(-1)  # 不可见步置零
        N, H, D = a_embs.size()
        alpha = torch.sigmoid(self.tm_alpha).view(1, 1, D)  # [1,1,D]

        mem = []
        prev = torch.zeros(N, D, device=a_embs.device, dtype=a_embs.dtype)
        for t in range(H):
            prev = alpha.squeeze(1) * prev + (1 - alpha.squeeze(1)) * a_masked[:, t, :]  # [N,D]
            mem.append(prev)
        mem = torch.stack(mem, dim=1)  # [N,H,D]
        a_embs = a_embs + 0.0 * mem#消融

        num_all_agent = a_length.size(0)  # N1+...+Nb
        m_embs = self.mode_tokens.weight.unsqueeze(0).repeat_interleave(self.num_historical_steps, 0)  # [H,K,D]
        m_embs = m_embs.unsqueeze(1).repeat_interleave(num_all_agent, 1).reshape(-1,
                                                                                 self.hidden_dim)  # [H*(N1,...,Nb)*K,D]

        m_batch = data['agent']['batch'].unsqueeze(1).repeat_interleave(self.num_modes, 1)  # [(N1,...,Nb),K]
        m_position = data['agent']['position'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(
            self.num_modes, 2)  # [(N1,...,Nb),H,K,2]
        m_heading = data['agent']['heading'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(
            self.num_modes, 2)  # [(N1,...,Nb),H,K]
        m_valid_mask = data['agent']['visible_mask'][:, :self.num_historical_steps].unsqueeze(2).repeat_interleave(
            self.num_modes, 2)  # [(N1,...,Nb),H,K]

        # ALL EDGE
        # t2m edge
        t2m_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)  # [(N1,...,Nb)*H,2]
        t2m_position_m = m_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        t2m_heading_t = data['agent']['heading'].reshape(-1)  # [(N1,...,Nb)]
        t2m_heading_m = m_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
        t2m_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]  # [(N1,...,Nb),H]
        t2m_valid_mask_m = m_valid_mask.reshape(num_all_agent, -1)  # [(N1,...,Nb),H*K]
        t2m_valid_mask = t2m_valid_mask_t.unsqueeze(2) & t2m_valid_mask_m.unsqueeze(1)  # [(N1,...,Nb),H,H*K]
        t2m_edge_index = dense_to_sparse(t2m_valid_mask)[0]
        t2m_edge_index = t2m_edge_index[:, torch.floor(t2m_edge_index[1] / self.num_modes) >= t2m_edge_index[0]]
        t2m_edge_index = t2m_edge_index[:,
                         torch.floor(t2m_edge_index[1] / self.num_modes) - t2m_edge_index[0] <= self.duration]
        t2m_edge_vector = transform_point_to_local_coordinate(t2m_position_t[t2m_edge_index[0]],
                                                              t2m_position_m[t2m_edge_index[1]],
                                                              t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_length, t2m_edge_attr_theta = compute_angles_lengths_2D(t2m_edge_vector)
        t2m_edge_attr_heading = wrap_angle(t2m_heading_t[t2m_edge_index[0]] - t2m_heading_m[t2m_edge_index[1]])
        t2m_edge_attr_interval = t2m_edge_index[0] - torch.floor(t2m_edge_index[1] / self.num_modes)
        t2m_edge_attr_input = torch.stack(
            [t2m_edge_attr_length, t2m_edge_attr_theta, t2m_edge_attr_heading, t2m_edge_attr_interval], dim=-1)
        t2m_edge_attr_embs = self.t2m_emb_layer(input=t2m_edge_attr_input)

        # l2m edge
        l2m_position_l = data['lane']['position']  # [(M1,...,Mb),2]
        l2m_position_m = m_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        l2m_heading_l = data['lane']['heading']  # [(M1,...,Mb)]
        l2m_heading_m = m_heading.reshape(-1)  # [(N1,...,Nb)]
        l2m_batch_l = data['lane']['batch']  # [(M1,...,Mb)]
        l2m_batch_m = m_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(
            -1)  # [(N1,...,Nb)*H*K]
        l2m_valid_mask_l = data['lane']['visible_mask']  # [(M1,...,Mb)]
        l2m_valid_mask_m = m_valid_mask.reshape(-1)  # [(N1,...,Nb)*H*K]
        l2m_valid_mask = l2m_valid_mask_l.unsqueeze(1) & l2m_valid_mask_m.unsqueeze(0)  # [(M1,...,Mb),(N1,...,Nb)*H*K]
        l2m_valid_mask = drop_edge_between_samples(l2m_valid_mask, batch=(l2m_batch_l, l2m_batch_m))
        l2m_edge_index = dense_to_sparse(l2m_valid_mask)[0]
        l2m_edge_index = l2m_edge_index[:,
                         torch.norm(l2m_position_l[l2m_edge_index[0]] - l2m_position_m[l2m_edge_index[1]], p=2,
                                    dim=-1) < self.l2a_radius]
        l2m_edge_vector = transform_point_to_local_coordinate(l2m_position_l[l2m_edge_index[0]],
                                                              l2m_position_m[l2m_edge_index[1]],
                                                              l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_length, l2m_edge_attr_theta = compute_angles_lengths_2D(l2m_edge_vector)
        l2m_edge_attr_heading = wrap_angle(l2m_heading_l[l2m_edge_index[0]] - l2m_heading_m[l2m_edge_index[1]])
        l2m_edge_attr_input = torch.stack([l2m_edge_attr_length, l2m_edge_attr_theta, l2m_edge_attr_heading], dim=-1)
        l2m_edge_attr_embs = self.l2m_emb_layer(input=l2m_edge_attr_input)

        # mode edge
        # m2m_a_edge
        m2m_a_position = m_position.permute(1, 2, 0, 3).reshape(-1, 2)  # [H*K*(N1,...,Nb),2]
        m2m_a_heading = m_heading.permute(1, 2, 0).reshape(-1)  # [H*K*(N1,...,Nb)]
        m2m_a_batch = data['agent']['batch']  # [(N1,...,Nb)]
        m2m_a_valid_mask = m_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes,
                                                                 -1)  # [H*K,(N1,...,Nb)]
        m2m_a_valid_mask = m2m_a_valid_mask.unsqueeze(2) & m2m_a_valid_mask.unsqueeze(
            1)  # [H*K,(N1,...,Nb),(N1,...,Nb)]
        m2m_a_valid_mask = drop_edge_between_samples(m2m_a_valid_mask, m2m_a_batch)
        m2m_a_edge_index = dense_to_sparse(m2m_a_valid_mask)[0]
        m2m_a_edge_index = m2m_a_edge_index[:, m2m_a_edge_index[1] != m2m_a_edge_index[0]]
        m2m_a_edge_index = m2m_a_edge_index[:,
                           torch.norm(m2m_a_position[m2m_a_edge_index[1]] - m2m_a_position[m2m_a_edge_index[0]], p=2,
                                      dim=-1) < self.a2a_radius]
        m2m_a_edge_vector = transform_point_to_local_coordinate(m2m_a_position[m2m_a_edge_index[0]],
                                                                m2m_a_position[m2m_a_edge_index[1]],
                                                                m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_length, m2m_a_edge_attr_theta = compute_angles_lengths_2D(m2m_a_edge_vector)
        m2m_a_edge_attr_heading = wrap_angle(m2m_a_heading[m2m_a_edge_index[0]] - m2m_a_heading[m2m_a_edge_index[1]])
        m2m_a_edge_attr_input = torch.stack([m2m_a_edge_attr_length, m2m_a_edge_attr_theta, m2m_a_edge_attr_heading],
                                            dim=-1)
        m2m_a_edge_attr_embs = self.m2m_a_emb_layer(input=m2m_a_edge_attr_input)

        # m2m_h
        m2m_h_position = m_position.permute(2, 0, 1, 3).reshape(-1, 2)  # [K*(N1,...,Nb)*H,2]
        m2m_h_heading = m_heading.permute(2, 0, 1).reshape(-1)  # [K*(N1,...,Nb)*H]
        m2m_h_valid_mask = m_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)  # [K*(N1,...,Nb),H]
        m2m_h_valid_mask = m2m_h_valid_mask.unsqueeze(2) & m2m_h_valid_mask.unsqueeze(1)  # [K*(N1,...,Nb),H,H]
        m2m_h_edge_index = dense_to_sparse(m2m_h_valid_mask)[0]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] > m2m_h_edge_index[0]]
        m2m_h_edge_index = m2m_h_edge_index[:, m2m_h_edge_index[1] - m2m_h_edge_index[0] <= self.duration]
        m2m_h_edge_vector = transform_point_to_local_coordinate(m2m_h_position[m2m_h_edge_index[0]],
                                                                m2m_h_position[m2m_h_edge_index[1]],
                                                                m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_length, m2m_h_edge_attr_theta = compute_angles_lengths_2D(m2m_h_edge_vector)
        m2m_h_edge_attr_heading = wrap_angle(m2m_h_heading[m2m_h_edge_index[0]] - m2m_h_heading[m2m_h_edge_index[1]])
        m2m_h_edge_attr_interval = m2m_h_edge_index[0] - m2m_h_edge_index[1]
        m2m_h_edge_attr_input = torch.stack(
            [m2m_h_edge_attr_length, m2m_h_edge_attr_theta, m2m_h_edge_attr_heading, m2m_h_edge_attr_interval], dim=-1)
        m2m_h_edge_attr_embs = self.m2m_h_emb_layer(input=m2m_h_edge_attr_input)

        # m2m_s edge
        m2m_s_valid_mask = m_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)  # [H*(N1,...,Nb),K]
        m2m_s_valid_mask = m2m_s_valid_mask.unsqueeze(2) & m2m_s_valid_mask.unsqueeze(1)  # [H*(N1,...,Nb),K,K]
        m2m_s_edge_index = dense_to_sparse(m2m_s_valid_mask)[0]
        m2m_s_edge_index = m2m_s_edge_index[:, m2m_s_edge_index[0] != m2m_s_edge_index[1]]

        # ALL ATTENTION
        # t2m attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H,D]
        m_embs_t = self.t2m_attn_layer(x=[t_embs, m_embs], edge_index=t2m_edge_index,
                                       edge_attr=t2m_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # l2m attention
        m_embs_l = self.l2m_attn_layer(x=[l_embs, m_embs], edge_index=l2m_edge_index,
                                       edge_attr=l2m_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # if l_embs.numel() > 0:
        #     M, D = l_embs.shape
        #     seq = l_embs.unsqueeze(1).repeat(1, self.num_historical_steps, 1)  # [M,H,D]
        #     h0 = torch.zeros(1, M, D, device=l_embs.device, dtype=l_embs.dtype)
        #     out, _ = self.scene_gru(seq, h0)  # [M,H,D]
        #     l_embs = self.scene_ln(l_embs + out[:, -1, :])
        # --- Scene Context Memory（零初始化残差，逐渐学到稳态偏置）---
        if l_embs.numel() > 0:
            l_embs = l_embs + self.scene_mlp(l_embs)  # [M,D]   消融l_embs = l_embs + self.scene_mlp(l_embs)改为l_embs = l_embs + 0.0 * self.scene_mlp(l_embs)

        g_m = torch.sigmoid(self.fuse_gate_m((m_embs_t - m_embs_l).detach()))  # [*,1]
        m_embs = g_m * m_embs_t + (1.0 - g_m) * m_embs_l
        # m_embs = m_embs_t + m_embs_l
        # g_m = self.fuse_gate_m(torch.cat([m_embs_t, m_embs_l], dim=-1))  # [*,1]
        # m_embs = g_m * m_embs_t + (1.0 - g_m) * m_embs_l

        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,
                                                                                                                     1).reshape(
            -1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        # moda attention
        for i in range(self.num_attn_layers):
            # m2m_a
            m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes,
                                    self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H*K*(N1,...,Nb),D]
            m_embs = self.m2m_a_attn_layers[i](x=m_embs, edge_index=m2m_a_edge_index, edge_attr=m2m_a_edge_attr_embs)
            # m2m_h
            m_embs = m_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(
                1, 2, 0, 3).reshape(-1, self.hidden_dim)  # [K*(N1,...,Nb)*H,D]
            m_embs = self.m2m_h_attn_layers[i](x=m_embs, edge_index=m2m_h_edge_index, edge_attr=m2m_h_edge_attr_embs)
            # m2m_s
            m_embs = m_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps,
                                    self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
            m_embs = self.m2m_s_attn_layers[i](x=m_embs, edge_index=m2m_s_edge_index)
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,
                                                                                                                     1).reshape(
            -1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D]

        # generate traj
        traj_propose = self.traj_propose(m_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes,
                                                         self.num_future_steps, 2)  # [(N1,...,Nb),H,K,F,2]
        traj_propose = transform_traj_to_global_coordinate(traj_propose, m_position, m_heading)  # [(N1,...,Nb),H,K,F,2]

        # generate anchor
        proposal = traj_propose.detach()  # [(N1,...,Nb),H,K,F,2]

        n_batch = m_batch  # [(N1,...,Nb),K]
        n_position = proposal[:, :, :, self.num_future_steps // 2, :]  # [(N1,...,Nb),H,K,2]
        _, n_heading = compute_angles_lengths_2D(
            proposal[:, :, :, self.num_future_steps // 2, :] - proposal[:, :, :, self.num_future_steps // 2 - 1,
                                                               :])  # [(N1,...,Nb),H,K]
        n_valid_mask = m_valid_mask  # [(N1,...,Nb),H,K]

        proposal = transform_traj_to_local_coordinate(proposal, n_position, n_heading)  # [(N1,...,Nb),H,K,F,2]
        anchor = self.proposal_to_anchor(proposal.reshape(-1, self.num_future_steps * 2))  # [(N1,...,Nb)*H*K,D]
        n_embs = anchor  # [(N1,...,Nb)*H*K,D]

        # t2n edge
        t2n_position_t = data['agent']['position'][:, :self.num_historical_steps].reshape(-1, 2)  # [(N1,...,Nb)*H,2]
        t2n_position_n = n_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        t2n_heading_t = data['agent']['heading'].reshape(-1)  # [(N1,...,Nb)]
        t2n_heading_n = n_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
        t2n_valid_mask_t = data['agent']['visible_mask'][:, :self.num_historical_steps]  # [(N1,...,Nb),H]
        t2n_valid_mask_n = n_valid_mask.reshape(num_all_agent, -1)  # [(N1,...,Nb),H*K]
        t2n_valid_mask = t2n_valid_mask_t.unsqueeze(2) & t2n_valid_mask_n.unsqueeze(1)  # [(N1,...,Nb),H,H*K]
        t2n_edge_index = dense_to_sparse(t2n_valid_mask)[0]
        t2n_edge_index = t2n_edge_index[:, torch.floor(t2n_edge_index[1] / self.num_modes) >= t2n_edge_index[0]]
        t2n_edge_index = t2n_edge_index[:,
                         torch.floor(t2n_edge_index[1] / self.num_modes) - t2n_edge_index[0] <= self.duration]
        t2n_edge_vector = transform_point_to_local_coordinate(t2n_position_t[t2n_edge_index[0]],
                                                              t2n_position_n[t2n_edge_index[1]],
                                                              t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_length, t2n_edge_attr_theta = compute_angles_lengths_2D(t2n_edge_vector)
        t2n_edge_attr_heading = wrap_angle(t2n_heading_t[t2n_edge_index[0]] - t2n_heading_n[t2n_edge_index[1]])
        t2n_edge_attr_interval = t2n_edge_index[0] - torch.floor(
            t2n_edge_index[1] / self.num_modes) - self.num_future_steps // 2
        t2n_edge_attr_input = torch.stack(
            [t2n_edge_attr_length, t2n_edge_attr_theta, t2n_edge_attr_heading, t2n_edge_attr_interval], dim=-1)
        t2n_edge_attr_embs = self.t2m_emb_layer(input=t2n_edge_attr_input)

        # l2n edge
        l2n_position_l = data['lane']['position']  # [(M1,...,Mb),2]
        l2n_position_n = n_position.reshape(-1, 2)  # [(N1,...,Nb)*H*K,2]
        l2n_heading_l = data['lane']['heading']  # [(M1,...,Mb)]
        l2n_heading_n = n_heading.reshape(-1)  # [(N1,...,Nb)*H*K]
        l2n_batch_l = data['lane']['batch']  # [(M1,...,Mb)]
        l2n_batch_n = n_batch.unsqueeze(1).repeat_interleave(self.num_historical_steps, 1).reshape(
            -1)  # [(N1,...,Nb)*H*K]
        l2n_valid_mask_l = data['lane']['visible_mask']  # [(M1,...,Mb)]
        l2n_valid_mask_n = n_valid_mask.reshape(-1)  # [(N1,...,Nb)*H*K]
        l2n_valid_mask = l2n_valid_mask_l.unsqueeze(1) & l2n_valid_mask_n.unsqueeze(0)  # [(M1,...,Mb),(N1,...,Nb)*H*K]
        l2n_valid_mask = drop_edge_between_samples(l2n_valid_mask, batch=(l2n_batch_l, l2n_batch_n))
        l2n_edge_index = dense_to_sparse(l2n_valid_mask)[0]
        l2n_edge_index = l2n_edge_index[:,
                         torch.norm(l2n_position_l[l2n_edge_index[0]] - l2n_position_n[l2n_edge_index[1]], p=2,
                                    dim=-1) < self.l2a_radius]
        l2n_edge_vector = transform_point_to_local_coordinate(l2n_position_l[l2n_edge_index[0]],
                                                              l2n_position_n[l2n_edge_index[1]],
                                                              l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_length, l2n_edge_attr_theta = compute_angles_lengths_2D(l2n_edge_vector)
        l2n_edge_attr_heading = wrap_angle(l2n_heading_l[l2n_edge_index[0]] - l2n_heading_n[l2n_edge_index[1]])
        l2n_edge_attr_input = torch.stack([l2n_edge_attr_length, l2n_edge_attr_theta, l2n_edge_attr_heading], dim=-1)
        l2n_edge_attr_embs = self.l2m_emb_layer(input=l2n_edge_attr_input)

        # mode edge
        # n2n_a_edge
        n2n_a_position = n_position.permute(1, 2, 0, 3).reshape(-1, 2)  # [H*K*(N1,...,Nb),2]
        n2n_a_heading = n_heading.permute(1, 2, 0).reshape(-1)  # [H*K*(N1,...,Nb)]
        n2n_a_batch = data['agent']['batch']  # [(N1,...,Nb)]
        n2n_a_valid_mask = n_valid_mask.permute(1, 2, 0).reshape(self.num_historical_steps * self.num_modes,
                                                                 -1)  # [H*K,(N1,...,Nb)]
        n2n_a_valid_mask = n2n_a_valid_mask.unsqueeze(2) & n2n_a_valid_mask.unsqueeze(
            1)  # [H*K,(N1,...,Nb),(N1,...,Nb)]
        n2n_a_valid_mask = drop_edge_between_samples(n2n_a_valid_mask, n2n_a_batch)
        n2n_a_edge_index = dense_to_sparse(n2n_a_valid_mask)[0]
        n2n_a_edge_index = n2n_a_edge_index[:, n2n_a_edge_index[1] != n2n_a_edge_index[0]]
        n2n_a_edge_index = n2n_a_edge_index[:,
                           torch.norm(n2n_a_position[n2n_a_edge_index[1]] - n2n_a_position[n2n_a_edge_index[0]], p=2,
                                      dim=-1) < self.a2a_radius]
        n2n_a_edge_vector = transform_point_to_local_coordinate(n2n_a_position[n2n_a_edge_index[0]],
                                                                n2n_a_position[n2n_a_edge_index[1]],
                                                                n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_length, n2n_a_edge_attr_theta = compute_angles_lengths_2D(n2n_a_edge_vector)
        n2n_a_edge_attr_heading = wrap_angle(n2n_a_heading[n2n_a_edge_index[0]] - n2n_a_heading[n2n_a_edge_index[1]])
        n2n_a_edge_attr_input = torch.stack([n2n_a_edge_attr_length, n2n_a_edge_attr_theta, n2n_a_edge_attr_heading],
                                            dim=-1)
        n2n_a_edge_attr_embs = self.m2m_a_emb_layer(input=n2n_a_edge_attr_input)

        # n2n_h edge
        n2n_h_position = n_position.permute(2, 0, 1, 3).reshape(-1, 2)  # [K*(N1,...,Nb)*H,2]
        n2n_h_heading = n_heading.permute(2, 0, 1).reshape(-1)  # [K*(N1,...,Nb)*H]
        n2n_h_valid_mask = n_valid_mask.permute(2, 0, 1).reshape(-1, self.num_historical_steps)  # [K*(N1,...,Nb),H]
        n2n_h_valid_mask = n2n_h_valid_mask.unsqueeze(2) & n2n_h_valid_mask.unsqueeze(1)  # [K*(N1,...,Nb),H,H]
        n2n_h_edge_index = dense_to_sparse(n2n_h_valid_mask)[0]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] > n2n_h_edge_index[0]]
        n2n_h_edge_index = n2n_h_edge_index[:, n2n_h_edge_index[1] - n2n_h_edge_index[0] <= self.duration]
        n2n_h_edge_vector = transform_point_to_local_coordinate(n2n_h_position[n2n_h_edge_index[0]],
                                                                n2n_h_position[n2n_h_edge_index[1]],
                                                                n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_length, n2n_h_edge_attr_theta = compute_angles_lengths_2D(n2n_h_edge_vector)
        n2n_h_edge_attr_heading = wrap_angle(n2n_h_heading[n2n_h_edge_index[0]] - n2n_h_heading[n2n_h_edge_index[1]])
        n2n_h_edge_attr_interval = n2n_h_edge_index[0] - n2n_h_edge_index[1]
        n2n_h_edge_attr_input = torch.stack(
            [n2n_h_edge_attr_length, n2n_h_edge_attr_theta, n2n_h_edge_attr_heading, n2n_h_edge_attr_interval], dim=-1)
        n2n_h_edge_attr_embs = self.m2m_h_emb_layer(input=n2n_h_edge_attr_input)

        # n2n_s edge
        n2n_s_position = n_position.transpose(0, 1).reshape(-1, 2)  # [H*(N1,...,Nb)*K,2]
        n2n_s_heading = n_heading.transpose(0, 1).reshape(-1)  # [H*(N1,...,Nb)*K]
        n2n_s_valid_mask = n_valid_mask.transpose(0, 1).reshape(-1, self.num_modes)  # [H*(N1,...,Nb),K]
        n2n_s_valid_mask = n2n_s_valid_mask.unsqueeze(2) & n2n_s_valid_mask.unsqueeze(1)  # [H*(N1,...,Nb),K,K]
        n2n_s_edge_index = dense_to_sparse(n2n_s_valid_mask)[0]
        n2n_s_edge_index = n2n_s_edge_index[:, n2n_s_edge_index[0] != n2n_s_edge_index[1]]
        n2n_s_edge_vector = transform_point_to_local_coordinate(n2n_s_position[n2n_s_edge_index[0]],
                                                                n2n_s_position[n2n_s_edge_index[1]],
                                                                n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_length, n2n_s_edge_attr_theta = compute_angles_lengths_2D(n2n_s_edge_vector)
        n2n_s_edge_attr_heading = wrap_angle(n2n_s_heading[n2n_s_edge_index[0]] - n2n_s_heading[n2n_s_edge_index[1]])
        n2n_s_edge_attr_input = torch.stack([n2n_s_edge_attr_length, n2n_s_edge_attr_theta, n2n_s_edge_attr_heading],
                                            dim=-1)
        n2n_s_edge_attr_embs = self.m2m_s_emb_layer(input=n2n_s_edge_attr_input)

        # t2n attention
        t_embs = a_embs.reshape(-1, self.hidden_dim)  # [(N1,...,Nb)*H,D]
        n_embs_t = self.t2n_attn_layer(x=[t_embs, n_embs], edge_index=t2n_edge_index,
                                       edge_attr=t2n_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # l2m attention
        n_embs_l = self.l2n_attn_layer(x=[l_embs, n_embs], edge_index=l2n_edge_index,
                                       edge_attr=l2n_edge_attr_embs)  # [(N1,...,Nb)*H*K,D]

        # n_embs = n_embs_t + n_embs_l
        g_n = torch.sigmoid(self.fuse_gate_n((n_embs_t - n_embs_l).detach()))  # [*,1]
        n_embs = g_n * n_embs_t + (1.0 - g_n) * n_embs_l
        # g_n = self.fuse_gate_n(torch.cat([n_embs_t, n_embs_l], dim=-1))  # [*,1]
        # n_embs = g_n * n_embs_t + (1.0 - g_n) * n_embs_l

        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0,
                                                                                                                     1).reshape(
            -1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        # moda attention
        for i in range(self.num_attn_layers):
            # m2m_a
            n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes,
                                    self.hidden_dim).transpose(1, 2).reshape(-1, self.hidden_dim)  # [H*K*(N1,...,Nb),D]
            n_embs = self.n2n_a_attn_layers[i](x=n_embs, edge_index=n2n_a_edge_index, edge_attr=n2n_a_edge_attr_embs)
            # m2m_h
            n_embs = n_embs.reshape(self.num_historical_steps, self.num_modes, num_all_agent, self.hidden_dim).permute(
                1, 2, 0, 3).reshape(-1, self.hidden_dim)  # [K*(N1,...,Nb)*H,D]
            n_embs = self.n2n_h_attn_layers[i](x=n_embs, edge_index=n2n_h_edge_index, edge_attr=n2n_h_edge_attr_embs)
            # m2m_s
            n_embs = n_embs.reshape(self.num_modes, num_all_agent, self.num_historical_steps,
                                    self.hidden_dim).transpose(0, 2).reshape(-1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
            n_embs = self.n2n_s_attn_layers[i](x=n_embs, edge_index=n2n_s_edge_index, edge_attr=n2n_s_edge_attr_embs)
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0,
                                                                                                                     1).reshape(
            -1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D

        # generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes,
                                                       self.num_future_steps, 2)  # [(N1,...,Nb),H,K,F,2]
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position,
                                                          n_heading)  # [(N1,...,Nb),H,K,F,2]

        return traj_propose, traj_output  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]