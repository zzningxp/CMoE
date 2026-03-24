from typing import Dict, Tuple

import numpy as np
import torch


QK_D = 128
BLOCK_BYTES_Q3_D = 52   # [d:fp16(2)][m:fp16(2)][ql:32][qh:16]
BLOCK_BYTES_Q4_D = 68   # [d:fp16(2)][m:fp16(2)][qs:64]
BLOCK_BYTES_Q5_D = 84   # [d:fp16(2)][m:fp16(2)][qh:16][qs:64]
BLOCK_BYTES_Q6_D = 100  # [d:fp16(2)][m:fp16(2)][ql:64][qh:32]


def _prepare_common(
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_int: torch.Tensor,
    groupsize: int,
    maxq: int,
) -> tuple[int, int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
    if groupsize != QK_D:
        raise ValueError(f"groupsize must be exactly {QK_D} for Q*_D packing")
    if q_int.ndim != 2:
        raise ValueError(f"q_int must be 2D, got shape={tuple(q_int.shape)}")

    out_features, in_features = q_int.shape
    if in_features % QK_D != 0:
        raise ValueError(f"in_features must be a multiple of {QK_D}, got {in_features}")

    num_groups = in_features // groupsize
    expected_shape = (out_features, num_groups)
    if tuple(scale.shape) != expected_shape:
        raise ValueError(f"scale shape mismatch, expected {expected_shape}, got {tuple(scale.shape)}")
    if tuple(zero.shape) != expected_shape:
        raise ValueError(f"zero shape mismatch, expected {expected_shape}, got {tuple(zero.shape)}")

    q_int = q_int.to(torch.uint8)
    if torch.any(q_int > maxq):
        raise ValueError(f"q_int contains values > {maxq}")

    n_blocks = in_features // QK_D
    if n_blocks != num_groups:
        raise ValueError(f"block/group mismatch: n_blocks={n_blocks}, num_groups={num_groups}")

    d_blocks = scale.to(torch.float16).contiguous()
    m_blocks = (-zero.to(torch.float32) * scale.to(torch.float32)).to(torch.float16).contiguous()
    q = q_int.reshape(out_features, n_blocks, QK_D).contiguous()
    return out_features, in_features, n_blocks, q, d_blocks, m_blocks


def _pack_bits(values: torch.Tensor, bits_per_value: int) -> torch.Tensor:
    if bits_per_value not in (1, 2):
        raise ValueError(f"Unsupported bits_per_value={bits_per_value}")
    items_per_byte = 8 // bits_per_value
    if values.shape[-1] % items_per_byte != 0:
        raise ValueError("values length is not aligned for packing")

    new_shape = (*values.shape[:-1], values.shape[-1] // items_per_byte, items_per_byte)
    vals = values.reshape(new_shape).to(torch.int32)
    shifts = torch.arange(0, 8, bits_per_value, device=values.device, dtype=torch.int32)
    shift_shape = (1,) * (vals.ndim - 1) + (items_per_byte,)
    packed = torch.sum(vals << shifts.view(shift_shape), dim=-1).to(torch.uint8)
    return packed


def _pack_block_rows(d_blocks: torch.Tensor, m_blocks: torch.Tensor, *payload_tensors: torch.Tensor) -> torch.Tensor:
    out_features, n_blocks = d_blocks.shape
    d_bytes = d_blocks.cpu().numpy().view(np.uint8).reshape(out_features, n_blocks, 2)
    m_bytes = m_blocks.cpu().numpy().view(np.uint8).reshape(out_features, n_blocks, 2)
    payload = [t.cpu().numpy().astype(np.uint8) for t in payload_tensors]
    packed = np.concatenate([d_bytes, m_bytes, *payload], axis=2)
    return torch.from_numpy(packed.reshape(out_features, -1).copy())


def pack_q4_d(
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_int: torch.Tensor,
    groupsize: int = 128,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    """
    Pack GPTQ tensors into Q4_D raw block bytes (QK_D = 128).

    Inputs:
    - scale: (out_features, num_groups), float16/float32
    - zero:  (out_features, num_groups), float16/float32
    - q_int: (out_features, in_features), uint8 values in [0, 15]
    """
    out_features, in_features, n_blocks, q, d_blocks, m_blocks = _prepare_common(
        scale=scale, zero=zero, q_int=q_int, groupsize=groupsize, maxq=15,
    )
    qs = ((q[:, :, 0::2] & 0x0F) | ((q[:, :, 1::2] & 0x0F) << 4)).to(torch.uint8).contiguous()
    packed_tensor = _pack_block_rows(d_blocks, m_blocks, qs)

    meta = {
        "out_features": out_features,
        "in_features": in_features,
        "num_groups": n_blocks,
        "n_blocks": n_blocks,
        "block_size": QK_D,
        "block_bytes": BLOCK_BYTES_Q4_D,
    }
    return packed_tensor, meta


def pack_q3_d(
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_int: torch.Tensor,
    groupsize: int = 128,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    out_features, in_features, n_blocks, q, d_blocks, m_blocks = _prepare_common(
        scale=scale, zero=zero, q_int=q_int, groupsize=groupsize, maxq=7,
    )
    ql = _pack_bits(q & 0x03, bits_per_value=2).contiguous()          # (..., 32)
    qh = _pack_bits((q >> 2) & 0x01, bits_per_value=1).contiguous()   # (..., 16)
    packed_tensor = _pack_block_rows(d_blocks, m_blocks, ql, qh)
    meta = {
        "out_features": out_features,
        "in_features": in_features,
        "num_groups": n_blocks,
        "n_blocks": n_blocks,
        "block_size": QK_D,
        "block_bytes": BLOCK_BYTES_Q3_D,
    }
    return packed_tensor, meta


def pack_q5_d(
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_int: torch.Tensor,
    groupsize: int = 128,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    out_features, in_features, n_blocks, q, d_blocks, m_blocks = _prepare_common(
        scale=scale, zero=zero, q_int=q_int, groupsize=groupsize, maxq=31,
    )
    qs = ((q[:, :, 0::2] & 0x0F) | ((q[:, :, 1::2] & 0x0F) << 4)).to(torch.uint8).contiguous()
    qh = _pack_bits((q >> 4) & 0x01, bits_per_value=1).contiguous()   # (..., 16)
    packed_tensor = _pack_block_rows(d_blocks, m_blocks, qh, qs)
    meta = {
        "out_features": out_features,
        "in_features": in_features,
        "num_groups": n_blocks,
        "n_blocks": n_blocks,
        "block_size": QK_D,
        "block_bytes": BLOCK_BYTES_Q5_D,
    }
    return packed_tensor, meta


def pack_q6_d(
    scale: torch.Tensor,
    zero: torch.Tensor,
    q_int: torch.Tensor,
    groupsize: int = 128,
) -> Tuple[torch.Tensor, Dict[str, int]]:
    out_features, in_features, n_blocks, q, d_blocks, m_blocks = _prepare_common(
        scale=scale, zero=zero, q_int=q_int, groupsize=groupsize, maxq=63,
    )
    ql = ((q[:, :, 0::2] & 0x0F) | ((q[:, :, 1::2] & 0x0F) << 4)).to(torch.uint8).contiguous()  # (..., 64)
    qh = _pack_bits((q >> 4) & 0x03, bits_per_value=2).contiguous()                               # (..., 32)
    packed_tensor = _pack_block_rows(d_blocks, m_blocks, ql, qh)
    meta = {
        "out_features": out_features,
        "in_features": in_features,
        "num_groups": n_blocks,
        "n_blocks": n_blocks,
        "block_size": QK_D,
        "block_bytes": BLOCK_BYTES_Q6_D,
    }
    return packed_tensor, meta


PACK_FUNCTIONS = {
    3: pack_q3_d,
    4: pack_q4_d,
    5: pack_q5_d,
    6: pack_q6_d,
}
