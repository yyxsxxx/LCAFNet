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
    输出是 gate 的 logits（形状 [...,1]），外面再 sigmoid
    """

    def __init__(self, input_dim: int, gate_hidden: int = None):
        super().__init__()
        if gate_hidden is None:
            gate_hidden = max(16, input_dim // 4)

        self.norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, gate_hidden)
        self.fc2 = nn.Linear(gate_hidden, 1)
        self.act = nn.GELU()

        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=0.1)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        h = self.norm(x)
        h = self.act(self.fc1(h))
        return self.fc2(h)  # logits


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
                 dropout: float,
                 # 新增：消融实验参数
                 use_fixed_fusion: bool = False,      # 是否使用固定融合
                 fixed_fusion_weight: float = 1,    # 固定权重值
                 ) -> None:
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
        
        # 消融实验参数
        self.use_fixed_fusion = use_fixed_fusion
        self.fixed_fusion_weight = fixed_fusion_weight

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

        # 场景MLP和门控融合
        self.scene_mlp = SceneMLPBlock(hidden_dim, expansion=4, dropout=0.1)
        self.fuse_gate_m = GatedMLP1D(input_dim=2 * hidden_dim, gate_hidden=None)
        self.fuse_gate_n = GatedMLP1D(input_dim=2 * hidden_dim, gate_hidden=None)
        
        # 如果使用固定融合，冻结门控网络的梯度
        if use_fixed_fusion:
            for param in self.fuse_gate_m.parameters():
                param.requires_grad = False
            for param in self.fuse_gate_n.parameters():
                param.requires_grad = False
        
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

        # 场景上下文记忆
        if l_embs.numel() > 0:
            l_embs = l_embs + self.scene_mlp(l_embs)  # [M,D]
        
        # ============ 第一阶段融合（m_embs） ============
        if self.use_fixed_fusion:
            # 确保门控网络参数被使用（解决DDP报错的关键）
            dummy_input_m = torch.zeros_like(torch.cat([m_embs_t, m_embs_l], dim=-1)[:1])
            _ = self.fuse_gate_m(dummy_input_m)  # 调用但不使用输出
            
            # 使用固定权重
            g_m = torch.full((m_embs_t.shape[0], 1), 
                            self.fixed_fusion_weight,
                            device=m_embs_t.device, 
                            dtype=m_embs_t.dtype)
        else:
            # 原始的门控融合
            gate_in_m = torch.cat([m_embs_t, m_embs_l], dim=-1).detach()  # [*, 2D]
            g_m = torch.sigmoid(self.fuse_gate_m(gate_in_m))              # [*, 1]
        
        m_embs = g_m * m_embs_t + (1.0 - g_m) * m_embs_l

        m_embs = m_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(
            -1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        
        # mode attention
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
        
        m_embs = m_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(
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
        
        # ============ 第二阶段融合（n_embs） ============
        if self.use_fixed_fusion:
            # 确保门控网络参数被使用
            dummy_input_n = torch.zeros_like(torch.cat([n_embs_t, n_embs_l], dim=-1)[:1])
            _ = self.fuse_gate_n(dummy_input_n)  # 调用但不使用输出
            
            g_n = torch.full((n_embs_t.shape[0], 1), 
                             self.fixed_fusion_weight,
                             device=n_embs_t.device, 
                             dtype=n_embs_t.dtype)
        else:
            # 原始的门控融合
            gate_in_n = torch.cat([n_embs_t, n_embs_l], dim=-1).detach()  # [*, 2D]
            g_n = torch.sigmoid(self.fuse_gate_n(gate_in_n))              # [*, 1]
        
        n_embs = g_n * n_embs_t + (1.0 - g_n) * n_embs_l

        n_embs = n_embs.reshape(num_all_agent, self.num_historical_steps, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(
            -1, self.hidden_dim)  # [H*(N1,...,Nb)*K,D]
        
        # mode attention
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
        
        n_embs = n_embs.reshape(self.num_historical_steps, num_all_agent, self.num_modes, self.hidden_dim).transpose(0, 1).reshape(
            -1, self.hidden_dim)  # [(N1,...,Nb)*H*K,D

        # generate refinement
        traj_refine = self.traj_refine(n_embs).reshape(num_all_agent, self.num_historical_steps, self.num_modes,
                                                       self.num_future_steps, 2)  # [(N1,...,Nb),H,K,F,2]
        traj_output = transform_traj_to_global_coordinate(proposal + traj_refine, n_position,
                                                          n_heading)  # [(N1,...,Nb),H,K,F,2]

        return traj_propose, traj_output  # [(N1,...,Nb),H,K,F,2],[(N1,...,Nb),H,K,F,2]