import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from typing import Optional, Tuple, List


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str = "silu"
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = F.silu if hidden_act == "silu" else getattr(F, hidden_act)

    def forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)

        intermediate = gate * up
        output = self.down_proj(intermediate)
        return output

class Router(nn.Module):
    def __init__(self, hidden_size, n_experts, n_activated, bias_speed = 0.001):
        super().__init__()
        self.dim = hidden_size
        self.topk = n_activated

        self.act_fn = F.silu
        self.gate = nn.Linear(n_experts, hidden_size, bias=False).to(torch.bfloat16)
        self.classifier = nn.Linear(n_experts, hidden_size, bias=False).to(torch.bfloat16)

        # self.extra_scale = nn.Parameter(torch.ones(n_experts, dtype=torch.bfloat16))
        # self.extra_bias = nn.Parameter(torch.zeros(n_experts, dtype=torch.float32))
        # self.bias_update_speed = bias_speed
    
    def update_bias(self, counts):
        mean_load = counts.mean()
        # Decrease bias for overloaded experts, increase for underloaded
        overloaded = counts > mean_load
        underloaded = counts < mean_load

        # self.extra_bias.data[overloaded] -= self.bias_update_speed
        # self.extra_bias.data[underloaded] += self.bias_update_speed

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.classifier is None:
            # scores = self.act_fn(self.gate(x)).abs()
            scores = self.gate(x)
        else:
            scores = (self.classifier(x) * self.act_fn(self.gate(x))).abs() 
            # scores = self.gate(x)

        # print(scores.shape)
        scores = scores.softmax(dim=-1, dtype=torch.float32)
        # scores = scores + self.extra_bias.to(x.device)[None, :]

        weights, indices = torch.topk(scores, self.topk, dim=-1)

        # original_scores = 1 + original_scores*self.extra_scale.to(x.device)
        # weights = original_scores.gather(1, indices)

        return weights.type_as(x), indices

class MoE(nn.Module):
    def __init__(self, hidden_size, moe_inter_dim, n_experts, n_shared, n_activated, return_router_info = False):
        super().__init__()
        self.dim = hidden_size
        n_routed_experts = n_experts - n_shared
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated
        self.experts_start_idx = 0
        self.experts_end_idx = n_routed_experts
        # self.gate = Router(hidden_size, n_routed_experts, self.n_activated_experts)
        self.gate = None
        self.n_shared_experts = n_shared
        # self.experts = nn.ModuleList([LlamaMLP(self.dim, moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
        #                               for i in range(self.n_routed_experts)])
        self.experts = None
        # self.shared_experts = LlamaMLP(self.dim, self.n_shared_experts * moe_inter_dim)
        self.shared_experts = None
        self.enable_scale = True
        self.return_router_info = return_router_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices = self.gate(x)
        # print(weights.shape, indices.shape)
        # print(weights[0])
        # print(indices[0])

        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts)

        counts = counts.tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            if self.enable_scale:
                y[idx] += expert(x[idx]) * weights[idx, top, None]
            else:
                y[idx] += expert(x[idx]) 
        if self.shared_experts is not None:
            z = self.shared_experts(x)
            hidden_states = (y + z).view(shape)
        else:
            hidden_states = y.view(shape)

        if self.return_router_info:
            return hidden_states, weights
        else:
            return hidden_states