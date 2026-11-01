#!/usr/bin/env python3
from contextlib import nullcontext
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Dict, Union
import argparse
import csv
import hashlib
import json
import math
import random
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from e3nn import o3, nn
from e3nn.o3 import Irreps
from e3nn.io import CartesianTensor
from e3nn.nn import FullyConnectedNet, Gate
from e3nn.math import soft_one_hot_linspace, soft_unit_step
from e3nn.util.jit import compile_mode
from tqdm import tqdm

try:
    from torch_scatter import scatter as _scatter
except ImportError:
    _scatter = None


# ============================================================================
# 1. 配置与模型构造
# ============================================================================


@dataclass
class TrainConfig:
    data_path: str = "4201350.json"
    limit: int = 0
    seed: int = 42
    batch_size: int = 16
    epochs: int = 30
    num_heads: int = 2
    lmax: int = 3
    embedding_dim: int = 64
    mul: int = 32
    layers: int = 2
    number_of_basis: int = 10
    radial_cutoff: float = 5.0
    max_radius: float = 7.0
    query_irreps: str = "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o"
    key_irreps: str = "32x0e+32x0o+16x1e+16x1o+8x2e+8x2o+4x3e+4x3o+2x4e+2x4o"
    output_irreps: str = "8x1o+4x2o+2x3o"
    formula: str = "ijk=ikj"
    test_size: float = 0.1
    split_seed: int = 42
    learning_rate: float = 0.005
    lr_gamma: float = 0.99
    grad_clip: float = 1.0
    device: str = "auto"
    check_finite: bool = False


def build_model(config, feature_dim, num_nodes):
    return Network(
        irreps_in=f"{feature_dim}x0e",
        embedding_dim=config.embedding_dim,
        irreps_query=config.query_irreps,
        irreps_key=config.key_irreps,
        irreps_out=config.output_irreps,
        formula=config.formula,
        max_radius=config.max_radius,
        num_nodes=num_nodes,
        mul=config.mul,
        layers=config.layers,
        number_of_basis=config.number_of_basis,
        lmax=config.lmax,
        num_heads=config.num_heads,
        pool_nodes=True,
        check_finite=config.check_finite,
    )


# ============================================================================
# 2. 分组 softmax 和稀疏聚合
# ============================================================================


def scatter(src, index, dim=0, dim_size=None, reduce="sum"):
    """Use torch-scatter when installed, otherwise native PyTorch index_add."""
    if _scatter is not None:
        return _scatter(src, index, dim=dim, dim_size=dim_size, reduce=reduce)
    if dim != 0 or reduce not in ("sum", "add", "mean"):
        raise ValueError("Native fallback supports dim=0 and sum/add/mean")
    size = (
        dim_size
        if dim_size is not None
        else int(index.max()) + 1
        if index.numel()
        else 0
    )
    out = src.new_zeros((size,) + src.shape[1:])
    out.index_add_(0, index, src)
    if reduce == "mean":
        count = src.new_zeros(size)
        count.index_add_(0, index, src.new_ones(index.numel()))
        out = out / count.clamp_min(1).reshape((-1,) + (1,) * (src.ndim - 1))
    return out


def segment_softmax(scores, edge_dst, num_nodes):
    """Softmax across incoming edges, independently for every head [E, H]."""
    output_dtype = scores.dtype
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    index = edge_dst[:, None].expand_as(scores)
    maxima = scores.new_full((num_nodes, scores.shape[1]), -float("inf"))
    maxima.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
    exp = (scores - maxima[edge_dst]).exp()
    denominator = scatter(exp, edge_dst, dim_size=num_nodes)
    return (exp / denominator[edge_dst]).to(output_dtype)


def attention_reduce(scores, values, edge_dst, cutoff, num_nodes):
    """Preserve cutoff renormalization and sqrt(alpha) value weighting."""
    probabilities = segment_softmax(scores, edge_dst, num_nodes)
    numerator = cutoff[:, None] * probabilities
    denominator = scatter(numerator, edge_dst, dim_size=num_nodes)
    denominator = torch.where(
        denominator == 0, torch.ones_like(denominator), denominator
    )
    alpha = numerator / denominator[edge_dst]
    return scatter(
        alpha.relu().sqrt().unsqueeze(-1) * values, edge_dst, dim_size=num_nodes
    )


# ============================================================================
# 3. 等变层、张量积与多头布局
# ============================================================================


class EquivariantLayerNormFast(torch.nn.Module):
    def __init__(self, irreps, eps=1e-05, affine=True, normalization="component"):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.eps = eps
        self.affine = affine
        num_scalar = sum((mul for (mul, ir) in self.irreps if ir.l == 0 and ir.p == 1))
        num_features = self.irreps.num_irreps
        if affine:
            self.affine_weight = torch.nn.Parameter(torch.ones(num_features))
            self.affine_bias = torch.nn.Parameter(torch.zeros(num_scalar))
        else:
            self.register_parameter("affine_weight", None)
            self.register_parameter("affine_bias", None)
        assert normalization in [
            "norm",
            "component",
        ], "normalization needs to be 'norm' or 'component'"
        self.normalization = normalization

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps}, eps={self.eps})"

    def forward(self, node_input, **kwargs):
        """
        Use torch layer norm for scalar features.
        """
        dim = node_input.shape[-1]
        fields = []
        ix = 0
        iw = 0
        ib = 0
        for (mul, ir) in self.irreps:
            d = ir.dim
            field = node_input.narrow(1, ix, mul * d)
            ix += mul * d
            if ir.l == 0 and ir.p == 1:
                weight = self.affine_weight[iw : iw + mul]
                bias = self.affine_bias[ib : ib + mul]
                iw += mul
                ib += mul
                field = F.layer_norm(field, tuple((mul,)), weight, bias, self.eps)
                fields.append(field.reshape(-1, mul * d))
                continue
            field = field.reshape(-1, mul, d)
            if self.normalization == "norm":
                field_norm = field.pow(2).sum(-1)
            elif self.normalization == "component":
                field_norm = field.pow(2).mean(-1)
            else:
                raise ValueError(
                    "Invalid normalization option {}".format(self.normalization)
                )
            field_norm = torch.mean(field_norm, dim=1, keepdim=True)
            field_norm = 1.0 / (field_norm + self.eps).sqrt()
            if self.affine:
                weight = self.affine_weight[None, iw : iw + mul]
                iw += mul
                field_norm = field_norm * weight
            field = field * field_norm.reshape(-1, mul, 1)
            fields.append(field.reshape(-1, mul * d))
        assert ix == dim
        output = torch.cat(fields, dim=-1)
        return output


@compile_mode("script")
class Vec2AttnHeads(torch.nn.Module):
    """
    Reshape vectors of shape [N, irreps_mid] to vectors of shape
    [N, num_heads, irreps_head].
    """

    def __init__(self, irreps_head, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.irreps_head = irreps_head
        self.irreps_mid_in = []
        for (mul, ir) in irreps_head:
            self.irreps_mid_in.append((mul * num_heads, ir))
        self.irreps_mid_in = o3.Irreps(self.irreps_mid_in)
        self.mid_in_indices = []
        start_idx = 0
        for (mul, ir) in self.irreps_mid_in:
            self.mid_in_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        (N, _) = x.shape
        out = []
        for (ir_idx, (start_idx, end_idx)) in enumerate(self.mid_in_indices):
            temp = x.narrow(1, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, self.num_heads, -1)
            out.append(temp)
        out = torch.cat(out, dim=2)
        return out

    def __repr__(self):
        return "{}(irreps_head={}, num_heads={})".format(
            self.__class__.__name__, self.irreps_head, self.num_heads
        )


@compile_mode("script")
class AttnHeads2Vec(torch.nn.Module):
    """
    Convert vectors of shape [N, num_heads, irreps_head] into
    vectors of shape [N, irreps_head * num_heads].
    """

    def __init__(self, irreps_head):
        super().__init__()
        self.irreps_head = irreps_head
        self.head_indices = []
        start_idx = 0
        for (mul, ir) in self.irreps_head:
            self.head_indices.append((start_idx, start_idx + mul * ir.dim))
            start_idx = start_idx + mul * ir.dim

    def forward(self, x):
        (N, _, _) = x.shape
        out = []
        for (ir_idx, (start_idx, end_idx)) in enumerate(self.head_indices):
            temp = x.narrow(2, start_idx, end_idx - start_idx)
            temp = temp.reshape(N, -1)
            out.append(temp)
        out = torch.cat(out, dim=1)
        return out

    def __repr__(self):
        return "{}(irreps_head={})".format(self.__class__.__name__, self.irreps_head)


class ShiftedSoftPlus(torch.nn.Module):
    """
    Shifted softplus as defined in SchNet, NeurIPS 2017.

    :param beta: value for the a more general softplus, default = 1
    :param threshold: values above are linear function, default = 20
    """

    _log2: float

    def __init__(self, beta=1, threshold=20):
        super().__init__()
        self.softplus = torch.nn.Softplus(beta=beta, threshold=threshold)
        self._log2 = math.log(2.0)

    def forward(self, x):
        """
        Evaluate shifted softplus

        :param x: torch.Tensor, input
        :return: torch.Tensor, ssp(x)
        """
        return self.softplus(x) - self._log2


class UVUTensorProduct(torch.nn.Module):
    def __init__(
        self,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        node_attr: o3.Irreps,
        internal_and_share_weights: bool = False,
    ):
        """
        UVU tensor product.

        Args:
            irreps_in1: irreps of first input, with available keys in `DataKey`
            irreps_in2: input of second input, with available keys in `DataKey`
            irreps_out: output irreps, with available keys in `DataKey`
            internal_and_share_weights: whether to create weights for the tensor
                product, if `True` all `mlp_*` params are ignored and if `False`,
                they should be provided to create an MLP to transform some data to be
                used as the weight of the tensor product.

        """
        super().__init__()
        self.out = irreps_out
        self.node_attr = node_attr
        irreps_mid = []
        instructions = []
        for (i, (mul, ir_in1)) in enumerate(irreps_in1):
            for (j, (_, ir_in2)) in enumerate(irreps_in2):
                for ir_out in ir_in1 * ir_in2:
                    if ir_out in irreps_out or ir_out == o3.Irreps("0e"):
                        k = len(irreps_mid)
                        irreps_mid.append((mul, ir_out))
                        instructions.append((i, j, k, "uvu", True))
        irreps_mid = o3.Irreps(irreps_mid)
        assert (
            irreps_mid.dim > 0
        ), f"irreps_in1={irreps_in1} times irreps_in2={irreps_in2} produces no instructions in irreps_out={irreps_out}"
        (self.irreps_mid, permutation, _) = irreps_mid.sort()
        instructions = [
            (i_1, i_2, permutation[i_out], mode, train)
            for (i_1, i_2, i_out, mode, train) in instructions
        ]
        self.lin0 = o3.FullyConnectedTensorProduct(
            irreps_in1, self.node_attr, irreps_in1
        )
        self.tp = o3.TensorProduct(
            irreps_in1,
            irreps_in2,
            self.irreps_mid,
            instructions,
            internal_weights=internal_and_share_weights,
            shared_weights=internal_and_share_weights,
        )
        self.dropout2 = nn.Dropout(irreps=self.irreps_mid, p=0.2)
        self.lin = o3.FullyConnectedTensorProduct(
            self.irreps_mid, self.node_attr, self.out
        )
        self.sc = o3.FullyConnectedTensorProduct(irreps_in1, self.node_attr, self.out)

    def forward(
        self, data1: Tensor, data2: Tensor, data_weight: Tensor, data3: Tensor
    ) -> Tensor:
        node_feats = data1
        node_attrs = data3
        edge_attrs = data2
        node_sc = self.sc(node_feats, node_attrs)
        node_feats = self.lin0(node_feats, node_attrs)
        node_feats = self.tp(node_feats, edge_attrs, data_weight)
        node_feats = self.dropout2(node_feats)
        node_conv_out = self.lin(node_feats, node_attrs)
        node_feats = node_sc + node_conv_out
        return node_feats

    def forward_edges(self, node_features, node_attrs, edge_src, edge_attrs, weights):
        """Project each source node once, then gather for its outgoing edges.

        Only deterministic node-local tensor products move before the gather.
        The edge tensor product and dropout keep their original order/layout.
        """
        skip = self.sc(node_features, node_attrs)[edge_src]
        projected = self.lin0(node_features, node_attrs)[edge_src]
        messages = self.tp(projected, edge_attrs, weights)
        messages = self.dropout2(messages)
        return skip + self.lin(messages, node_attrs[edge_src])


def tp_path_exists(irreps_in1, irreps_in2, ir_out) -> bool:
    irreps_in1 = o3.Irreps(irreps_in1).simplify()
    irreps_in2 = o3.Irreps(irreps_in2).simplify()
    ir_out = o3.Irrep(ir_out)
    for (_, ir1) in irreps_in1:
        for (_, ir2) in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False


class Compose(torch.nn.Module):
    def __init__(self, first, second) -> None:
        super().__init__()
        self.first = first
        self.second = second

    def forward(self, *input):
        x = self.first(*input)
        return self.second(x)


# ============================================================================
# 4. 等变注意力
# ============================================================================


def multiheadsplit(x, num_heads=2):
    x = o3.Irreps(x)
    if any((mul % num_heads for (mul, _) in x)):
        raise ValueError("Every irrep multiplicity must be divisible by num_heads")
    ll = []
    for (mul, ir) in x:
        ll.append((int(mul // int(num_heads)), ir))
    return o3.Irreps(ll)


@compile_mode("script")
class Attention(torch.nn.Module):
    def __init__(
        self,
        node_attr,
        irreps_node_input,
        irreps_query,
        irreps_key,
        irreps_output,
        number_of_basis,
        num_heads=2,
        lmax=3,
    ):
        super().__init__()
        self.heads = num_heads
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.norm = EquivariantLayerNormFast(irreps=irreps_output)
        self.number_of_basis = number_of_basis
        self.irreps_query = irreps_query
        self.node_attr = node_attr
        self.h_q = o3.FullyConnectedTensorProduct(
            irreps_node_input, self.node_attr, irreps_query
        )
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_output = o3.Irreps(irreps_output)
        self.tp_k = UVUTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_key, self.node_attr
        )
        self.fc_k = FullyConnectedNet(
            self.number_of_basis + [self.tp_k.tp.weight_numel],
            act=torch.nn.functional.silu,
        )
        self.tp_v = UVUTensorProduct(
            self.irreps_node_input, self.irreps_sh, self.irreps_output, self.node_attr
        )
        self.fc_v = FullyConnectedNet(
            self.number_of_basis + [self.tp_v.tp.weight_numel],
            act=torch.nn.functional.silu,
        )
        self.dot = torch.nn.ModuleList(
            [
                o3.FullyConnectedTensorProduct(
                    multiheadsplit(self.irreps_query, self.heads).simplify(),
                    multiheadsplit(self.irreps_key, self.heads).simplify(),
                    "0e",
                )
                for _ in range(self.heads)
            ]
        )
        self.vec2headsq = Vec2AttnHeads(
            multiheadsplit(self.irreps_query, self.heads).simplify(), self.heads
        )
        self.vec2headsk = Vec2AttnHeads(
            multiheadsplit(self.irreps_key, self.heads).simplify(), self.heads
        )
        self.vec2headsv = Vec2AttnHeads(
            multiheadsplit(self.irreps_output, self.heads).simplify(), self.heads
        )
        self.heads2vecv = AttnHeads2Vec(
            multiheadsplit(self.irreps_output, self.heads).simplify()
        )
        self.lin = o3.FullyConnectedTensorProduct(
            self.irreps_output, self.node_attr, self.irreps_output
        )
        self.sc = o3.FullyConnectedTensorProduct(
            irreps_node_input, node_attr, self.irreps_output
        )
        self.dim = self.irreps_key.dim

    def forward(
        self,
        node_attr,
        node_input,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        edge_length,
        fpit=None,
    ) -> torch.Tensor:
        residual = self.sc(node_input, node_attr)
        q = self.vec2headsq(self.h_q(node_input, node_attr))[edge_dst]
        key_weights = self.fc_k(edge_scalars)
        k = self.tp_k.forward_edges(
            node_input, node_attr, edge_src, edge_attr, key_weights
        )
        value_weights = self.fc_v(edge_scalars)
        v = self.tp_v.forward_edges(
            node_input, node_attr, edge_src, edge_attr, value_weights
        )
        k = self.vec2headsk(k)
        v = self.vec2headsv(v)
        scores = torch.cat(
            [dot(q[:, i], k[:, i]) for (i, dot) in enumerate(self.dot)], dim=-1
        )
        scores = scores / self.dim**0.5
        messages = attention_reduce(
            scores, v, edge_dst, edge_length, node_input.shape[0]
        )
        return self.lin(self.heads2vecv(messages), node_attr) + residual


class EquivariantAttention(torch.nn.Module):
    def __init__(
        self,
        node_attr,
        irreps_node_input,
        irreps_query,
        irreps_key,
        irreps_node_hidden,
        irreps_node_output,
        irreps_edge_attr,
        layers,
        fc_neurons,
        num_heads=2,
        lmax=3,
    ) -> None:
        super().__init__()
        self.attr = o3.Irreps(node_attr)
        self.irreps_node_input = o3.Irreps(irreps_node_input)
        self.irreps_query = o3.Irreps(irreps_query)
        self.irreps_key = o3.Irreps(irreps_key)
        self.irreps_node_hidden = o3.Irreps(irreps_node_hidden)
        self.irreps_node_output = o3.Irreps(irreps_node_output)
        self.irreps_edge_attr = o3.Irreps(irreps_edge_attr)
        act = {1: torch.nn.functional.silu, -1: torch.tanh}
        act_gates = {1: ShiftedSoftPlus(), -1: torch.tanh}
        self.layers = torch.nn.ModuleList()
        for _ in range(layers):
            irreps_scalars = o3.Irreps(
                [
                    (mul, ir)
                    for (mul, ir) in self.irreps_node_hidden
                    if ir.l == 0
                    and tp_path_exists(
                        self.irreps_node_input, self.irreps_edge_attr, ir
                    )
                ]
            ).simplify()
            irreps_gated = o3.Irreps(
                [
                    (mul, ir)
                    for (mul, ir) in self.irreps_node_hidden
                    if ir.l > 0
                    and tp_path_exists(
                        self.irreps_node_input, self.irreps_edge_attr, ir
                    )
                ]
            )
            ir = (
                "0e"
                if tp_path_exists(self.irreps_node_input, self.irreps_edge_attr, "0e")
                else "0o"
            )
            irreps_gates = o3.Irreps(
                [(mul, ir) for (mul, _) in irreps_gated]
            ).simplify()
            gate = Gate(
                irreps_scalars,
                [act[ir.p] for (_, ir) in irreps_scalars],
                irreps_gates,
                [act_gates[ir.p] for (_, ir) in irreps_gates],
                irreps_gated,
            )
            conv = Attention(
                self.attr,
                self.irreps_node_input,
                self.irreps_query,
                self.irreps_key,
                gate.irreps_in,
                fc_neurons,
                num_heads=num_heads,
                lmax=lmax,
            )
            self.irreps_node_input = gate.irreps_out
            self.norm = EquivariantLayerNormFast(self.irreps_node_input)
            self.layers.append(Compose(Compose(conv, gate), self.norm))
        self.layers.append(
            Attention(
                self.attr,
                self.irreps_node_input,
                self.irreps_query,
                self.irreps_key,
                self.irreps_node_output,
                fc_neurons,
                num_heads=num_heads,
                lmax=lmax,
            )
        )

    def forward(
        self,
        node_attr,
        node_features,
        edge_src,
        edge_dst,
        edge_attr,
        edge_scalars,
        edge_length,
    ) -> torch.Tensor:
        for lay in self.layers:
            node_features = lay(
                node_attr,
                node_features,
                edge_src,
                edge_dst,
                edge_attr,
                edge_scalars,
                edge_length,
            )
        return node_features


# ============================================================================
# 5. 笛卡尔张量读出
# ============================================================================


class Fromtensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)

    def forward(self, data):
        return self.tensor.from_cartesian(data)


class Totensor(torch.nn.Module):
    def __init__(self, formula):
        super().__init__()
        self.tensor = CartesianTensor(formula)

    def forward(self, data):
        return self.tensor.to_cartesian(data)


class TensorIrreps(torch.nn.Module):
    def __init__(self, formula, conv_to_output_hidden_irreps_out):
        super().__init__()
        if formula is None:
            self.formula = formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            self.irreps_out = o3.Irreps("0e")
            self.extra_layers = o3.Linear(
                irreps_in=self.irreps_in, irreps_out=self.irreps_out
            )
        else:
            self.formula = formula
            self.irreps_in = conv_to_output_hidden_irreps_out
            self.irreps_out = CartesianTensor(formula=self.formula)
            self.extra_layers = o3.Linear(
                irreps_in=self.irreps_in, irreps_out=self.irreps_out
            )
            self.register_buffer(
                "_cartesian_basis",
                self.irreps_out.reduced_tensor_products().change_of_basis,
                persistent=False,
            )

    def forward(self, data):
        out = self.extra_layers(data)
        if self.formula is None:
            return out
        else:
            basis = self._cartesian_basis.to(out)
            return (out @ basis.flatten(1)).reshape(out.shape[:-1] + basis.shape[1:])


# ============================================================================
# 6. 完整周期图网络
# ============================================================================


@compile_mode("script")
class Network(torch.nn.Module):
    def __init__(
        self,
        irreps_in,
        embedding_dim,
        irreps_query,
        irreps_key,
        irreps_out,
        formula,
        max_radius,
        num_nodes,
        mul=32,
        layers=2,
        number_of_basis=10,
        lmax=3,
        pool_nodes=True,
        num_heads=2,
        check_finite=False,
    ) -> None:
        super().__init__()
        self.check_finite = check_finite
        self.lmax = lmax
        self.max_radius = max_radius
        self.number_of_basis = number_of_basis
        self.num_nodes = num_nodes
        self.pool_nodes = pool_nodes
        self.formula = formula
        self.irreps_in = irreps_in
        self.embeding_dim = embedding_dim
        irreps_node_hidden = o3.Irreps(
            [(int(mul / 2**l), (l, p)) for l in range(lmax + 1) for p in [-1, 1]]
        )
        self.irreps_node_hidden = irreps_node_hidden.simplify()
        self.irreps_query = irreps_query
        self.irreps_key = irreps_key
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax)
        self.lin = o3.Linear(self.irreps_in, "{}x0e".format(self.embeding_dim))
        self.GAT = EquivariantAttention(
            node_attr=self.irreps_in,
            irreps_node_input="{}x0e".format(self.embeding_dim),
            irreps_query=irreps_query,
            irreps_key=irreps_key,
            irreps_node_hidden=self.irreps_node_hidden,
            irreps_node_output=irreps_out,
            irreps_edge_attr=self.irreps_sh,
            layers=layers,
            fc_neurons=[self.number_of_basis, 100],
            num_heads=num_heads,
            lmax=lmax,
        )
        self.irreps_in = self.GAT.irreps_node_input
        self.irreps_out = self.GAT.irreps_node_output
        self.TI0 = o3.Linear(self.irreps_out, self.irreps_out)
        self.TI1 = o3.Linear(self.irreps_out, self.irreps_out)
        self.TI = TensorIrreps(self.formula, self.irreps_out)
        self.dropout1 = nn.Dropout(irreps=self.irreps_out, p=0.2)

    def preprocess(self, data) -> torch.Tensor:
        if "batch" in data:
            batch = data["batch"]
        else:
            batch = data["pos"].new_zeros(data["pos"].shape[0], dtype=torch.long)
        edge_src = data["edge_index"][0]
        edge_dst = data["edge_index"][1]
        edge_batch = batch[edge_src]
        edge_vec = (
            data["pos"][edge_dst]
            - data["pos"][edge_src]
            + torch.einsum(
                "ni,nij->nj", data["edge_shift"], data["lattice"][edge_batch]
            )
        )
        return (batch, data["x"], edge_src, edge_dst, edge_vec)

    def forward(self, data) -> torch.Tensor:
        (batch, node_inputs, edge_src, edge_dst, edge_vec) = self.preprocess(data)
        del data
        node_attr = node_inputs
        edge_attr = o3.spherical_harmonics(
            self.irreps_sh, edge_vec, True, normalization="component"
        )
        edge_length = edge_vec.norm(dim=1)
        edge_length_embedding = soft_one_hot_linspace(
            edge_length,
            0.0,
            self.max_radius,
            self.number_of_basis,
            basis="smooth_finite",
            cutoff=True,
        ).mul(self.number_of_basis**0.5)
        edge_weight_cutoff = soft_unit_step(10 * (1 - edge_length / self.max_radius))
        node_features = self.lin(node_inputs)
        node_outputs = self.GAT(
            node_attr,
            node_features,
            edge_src,
            edge_dst,
            edge_attr,
            edge_length_embedding,
            edge_weight_cutoff,
        )
        if self.pool_nodes:
            node_outputs = scatter(node_outputs, batch, dim=0, reduce="mean").div(
                self.num_nodes**0.5
            )
        else:
            pass
        node_outputs1 = self.TI0(node_outputs)
        node_outputs2 = self.TI1(node_outputs1)
        node_outputs = node_outputs2 + node_outputs
        node_outputs = self.TI(node_outputs)
        if self.check_finite and torch.isnan(node_outputs).any():
            print("nan after TI")
        return node_outputs


# ============================================================================
# 7. 元素特征、邻居图与数据划分
# ============================================================================


def element_features():
    from ase.atoms import Atom
    from jarvis.core.specie import Specie
    from sklearn.preprocessing import OneHotEncoder

    try:
        encoder = OneHotEncoder(max_categories=6, sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(max_categories=6, sparse=False)
    descriptions = [
        Specie(Atom(i).symbol, source="magpie").get_descrp_arr for i in range(1, 102)
    ]
    return encoder.fit_transform(descriptions)


def r_cut2D(radius, atoms):
    from pymatgen.io.ase import AseAtomsAdaptor

    cell = AseAtomsAdaptor.get_structure(atoms).lattice.matrix
    return max(np.linalg.norm(cell[0]), np.linalg.norm(cell[1]), radius)


def datatransform(crystal, target, features, radial_cutoff, device):
    import ase.neighborlist
    from ase.data import atomic_numbers
    from torch_geometric.data import Data

    (edge_src, edge_dst, edge_shift) = ase.neighborlist.neighbor_list(
        "ijS", a=crystal, cutoff=r_cut2D(radial_cutoff, crystal), self_interaction=False
    )
    return Data(
        pos=torch.as_tensor(
            crystal.get_positions(), dtype=torch.float32, device=device
        ),
        lattice=torch.as_tensor(
            crystal.cell.array, dtype=torch.float32, device=device
        ).unsqueeze(0),
        x=torch.as_tensor(
            np.asarray(
                [features[atomic_numbers[atom] - 1] for atom in crystal.symbols]
            ),
            dtype=torch.float32,
            device=device,
        ),
        edge_index=torch.as_tensor(
            np.stack([edge_src, edge_dst]), dtype=torch.long, device=device
        ),
        edge_shift=torch.as_tensor(edge_shift, dtype=torch.float32, device=device),
        energy=torch.as_tensor(target, dtype=torch.float32, device=device).unsqueeze(0),
    )


def build_loaders(config, device):
    import pandas as pd
    from pymatgen.core import Structure
    from pymatgen.io.ase import AseAtomsAdaptor
    from sklearn.model_selection import train_test_split
    from torch_geometric.loader import DataLoader

    frame = pd.read_json(config.data_path)
    if "structure" not in frame or "total" not in frame:
        raise ValueError(
            "Expected original training JSON: structure (pymatgen dictionaries) and total (Cartesian rank-3 tensors). Download envelopes require explicit conversion."
        )
    stop = None if config.limit == 0 else config.limit
    structures = [
        AseAtomsAdaptor.get_atoms(Structure.from_dict(s))
        for s in frame["structure"][:stop]
    ]
    targets = list(frame["total"][:stop])
    if len(structures) < 2:
        raise ValueError(
            "At least two selected structures are required for train/validation split"
        )
    expected = (3, 3, 3)
    for (i, target) in enumerate(targets):
        if np.asarray(target).shape != expected:
            raise ValueError(
                f"Target {i} has shape {np.asarray(target).shape}; formula {config.formula!r} requires {expected}. No implicit Voigt conversion is performed."
            )
    features = element_features()
    num_nodes = sum((len(s) for s in structures)) / len(structures)
    dataset = [
        datatransform(s, y, features, config.radial_cutoff, device)
        for (s, y) in zip(structures, targets)
    ]
    (train, valid) = train_test_split(
        dataset,
        test_size=config.test_size,
        random_state=config.split_seed,
        shuffle=True,
    )
    return (
        DataLoader(train, batch_size=config.batch_size),
        DataLoader(valid, batch_size=config.batch_size),
        features.shape[1],
        num_nodes,
    )


# ============================================================================
# 8. 训练与验证
# ============================================================================


def train(config, epoch_callback=None):
    torch.set_default_dtype(torch.float32)
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else torch.device(config.device)
    )
    (train_loader, valid_loader, feature_dim, num_nodes) = build_loaders(config, device)
    model_info = {
        "feature_dim": int(feature_dim),
        "num_nodes": float(num_nodes),
        "train_samples": len(train_loader.dataset)
        if hasattr(train_loader, "dataset")
        else None,
        "validation_samples": len(valid_loader.dataset)
        if hasattr(valid_loader, "dataset")
        else None,
    }
    net = build_model(config, feature_dim, num_nodes).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=config.learning_rate)
    loss_fn = torch.nn.MSELoss().to(device)
    scaler = GradScaler(enabled=device.type == "cuda")
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)
    history = []
    for epoch in range(config.epochs):
        net.train()
        train_loss = torch.zeros((), device=device)
        for batch in tqdm(train_loader, desc=f"train {epoch}"):
            optimizer.zero_grad()
            with autocast() if device.type == "cuda" else nullcontext():
                output = net(batch)
                loss = loss_fn(output, batch.energy)
            train_loss += loss.detach()
            if torch.isnan(loss):
                print("NaN detected in loss before backward; skipping this step")
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm_(net.parameters(), max_norm=config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        net.eval()
        valid_loss = torch.zeros((), device=device)
        mae = torch.zeros((), device=device)
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc=f"valid {epoch}"):
                with autocast() if device.type == "cuda" else nullcontext():
                    output = net(batch)
                    loss = loss_fn(output, batch.energy)
                valid_loss += loss
                mae += torch.mean(torch.abs(output - batch.energy))
        metrics = dict(
            epoch=epoch,
            train_loss=(train_loss / len(train_loader)).item(),
            valid_loss=(valid_loss / len(valid_loader)).item(),
            mae=(mae / len(valid_loader)).item(),
        )
        history.append(metrics)
        print(metrics)
        if epoch_callback is not None:
            epoch_callback(net, history, model_info)
    return (net, history)


# ============================================================================
# 9. 指标、模型保存和可选绘图
# ============================================================================


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    # Non-finite losses remain inspectable in interrupted/diverged runs without
    # emitting nonstandard JSON tokens. Successful runs contain ordinary floats.
    def clean(value):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    temporary.write_text(
        json.dumps(clean(payload), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_history_csv(path, history):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["epoch", "train_mse", "validation_mse", "validation_mae"])
        for row in history:
            writer.writerow(
                [row["epoch"] + 1, row["train_loss"], row["valid_loss"], row["mae"]]
            )


def plot_history(history, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = np.asarray([row["epoch"] + 1 for row in history])
    training = np.asarray([row["train_loss"] for row in history])
    validation = np.asarray([row["valid_loss"] for row in history])
    mae = np.asarray([row["mae"] for row in history])
    if not np.isfinite(np.concatenate([training, validation, mae])).all():
        raise ValueError(
            "Non-finite metrics recorded; inspect training.json before plotting"
        )
    with plt.rc_context(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    ):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
        for ax in axes[:2]:
            ax.plot(epochs, training, label="Training", color="#de8f05", linewidth=1.8)
            ax.plot(
                epochs, validation, label="Validation", color="#007c83", linewidth=1.8
            )
            ax.set_ylabel("MSE (original label scale)")
            ax.legend(frameon=False)
        axes[0].set_title("Training and validation MSE")
        axes[1].set_title("MSE on a logarithmic scale")
        if (training > 0).all() and (validation > 0).all():
            axes[1].set_yscale("log")
        axes[2].plot(epochs, mae, color="#007c83", linewidth=1.8, label="Validation")
        axes[2].set_title("Validation MAE")
        axes[2].set_ylabel("MAE (original label scale)")
        axes[2].legend(frameon=False)
        for ax in axes:
            ax.set_xlabel("Epoch")
            ax.set_xlim(0.5, max(1, int(epochs[-1])) + 0.5)
            ax.grid(alpha=0.2)
        fig.savefig(output_dir / "loss_curves.png", dpi=200)
        fig.savefig(output_dir / "loss_curves.pdf")
        plt.close(fig)


class RunRecorder:
    """Persist per-epoch metrics and the best validation-MSE state_dict."""

    def __init__(self, output_dir, config, metadata):
        self.output_dir = output_dir
        self.config = config
        self.metadata = metadata
        self.best_mse = float("inf")
        self.best_epoch = None

    def __call__(self, model, history, model_info):
        latest = history[-1]
        if math.isfinite(latest["valid_loss"]) and latest["valid_loss"] < self.best_mse:
            self.best_mse = latest["valid_loss"]
            self.best_epoch = latest["epoch"] + 1
            torch.save(model.state_dict(), self.output_dir / "best_model.pt")
        self.metadata.update(
            best_validation_mse=self.best_mse, best_epoch=self.best_epoch
        )
        atomic_json(
            self.output_dir / "training.json",
            {
                "config": asdict(self.config),
                "metadata": self.metadata,
                "model": model_info,
                "history": history,
            },
        )
        write_history_csv(self.output_dir / "loss_history.csv", history)


# ============================================================================
# 10. 命令行入口；只有直接执行文件时才开始训练
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="EATGNN training", formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config", type=Path, help="Optional JSON overrides for TrainConfig"
    )
    parser.add_argument(
        "--data", help="Original training JSON with structure and total columns"
    )
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--device", help="auto, cpu or cuda[:index]")
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--limit", type=int, help="0: all records; -1: original slicing behavior"
    )
    parser.add_argument("--output", type=Path, default=Path("runs/eatgnn_corrected"))
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Export MSE/MAE PNG and PDF using matplotlib",
    )
    parser.add_argument("--check-finite", action="store_true", default=None)
    args = parser.parse_args()
    settings = asdict(TrainConfig())
    if args.config:
        overrides = json.loads(args.config.read_text(encoding="utf-8"))
        if not isinstance(overrides, dict) or set(overrides) - set(settings):
            parser.error(
                "Config must be a JSON object containing only TrainConfig fields"
            )
        settings.update(overrides)
    for argument, key in [
        ("data", "data_path"),
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("learning_rate", "learning_rate"),
        ("device", "device"),
        ("seed", "seed"),
        ("limit", "limit"),
        ("check_finite", "check_finite"),
    ]:
        value = getattr(args, argument)
        if value is not None:
            settings[key] = value
    config = TrainConfig(**settings)
    data_path = Path(config.data_path).expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset does not exist: {data_path}")
    if (
        config.epochs < 1
        or config.batch_size < 1
        or config.learning_rate <= 0
        or not 0 < config.test_size < 1
    ):
        parser.error("Require epochs/batch-size/lr > 0 and 0 < test_size < 1")
    if not 0 <= config.seed < 2**32:
        parser.error("Seed must be between 0 and 2**32-1")
    if any(
        (args.output / name).exists()
        for name in ("training.json", "model.pt", "best_model.pt")
    ):
        parser.error(
            f"Existing experiment found in {args.output}; choose a new --output directory"
        )
    if args.plot:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            parser.error("--plot requires matplotlib: pip install matplotlib")
    config.data_path = str(data_path)
    args.output.mkdir(parents=True, exist_ok=True)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.set_num_threads(4)
    device = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if config.device == "auto"
        else config.device
    )
    metadata = {
        "seed": config.seed,
        "split_seed": config.split_seed,
        "device": device,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "amp": torch.device(device).type == "cuda",
        "gpu": torch.cuda.get_device_name(torch.device(device))
        if torch.device(device).type == "cuda"
        else None,
        "dataset_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "metric": "Unweighted mean of batch MSE/MAE, matching the original training script",
        "label_scale": "Unmodified source tensor labels; no unit or orientation conversion",
        "status": "running",
    }
    recorder = RunRecorder(args.output, config, metadata)
    start = time.perf_counter()
    try:
        model, history = train(config, epoch_callback=recorder)
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize(torch.device(device))
        metadata.update(status="complete", elapsed_seconds=time.perf_counter() - start)
        torch.save(model.state_dict(), args.output / "model.pt")
        payload = json.loads(
            (args.output / "training.json").read_text(encoding="utf-8")
        )
        payload["metadata"] = metadata
        atomic_json(args.output / "training.json", payload)
        if args.plot:
            plot_history(history, args.output)
        print(f"Saved experiment to: {args.output.resolve()}")
        print(
            f"Best validation MSE: {recorder.best_mse:.8g} (epoch {recorder.best_epoch})"
        )
    except BaseException:
        import traceback

        (args.output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
