# CUDAGraph wrapper for FP8+FP8 residual decode kernel.
from __future__ import annotations

from typing import Optional

import torch

from .attn_kernel_v1210_fused_bsz_fp8fp8 import (
    FP8FP8DecodeWorkspace,
    attn_forward_decode_fp8fp8,
)


class CUDAGraphDecodeRunnerFP8FP8:
    """Capture and replay the FP8+FP8 residual decode kernel with static buffers."""

    def __init__(
        self,
        q: torch.Tensor,
        k_fp8: torch.Tensor,
        v: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        seqlen: int | torch.Tensor | None = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        use_fp8_residual: bool = True,
        copy_kv: bool = True,
        workspace: Optional[FP8FP8DecodeWorkspace] = None,
        warmup: int = 2,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDAGraph capture.")
        if q.device.type != "cuda":
            raise ValueError("q must be a CUDA tensor.")

        self._device = q.device
        self._scale = scale
        self._BS = BS
        self._SBS = SBS
        self._delta = delta
        self._use_fp8_residual = use_fp8_residual
        self._use_ext_th = precomputed_threshold is not None
        self._copy_kv = copy_kv

        if self._use_fp8_residual and k_residual is None:
            raise ValueError("use_fp8_residual=True requires k_residual")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required when use_ext_th=True")
        if isinstance(seqlen, torch.Tensor):
            if seqlen.numel() != 1:
                raise ValueError("seqlen tensor must be a scalar")
            if seqlen.device != self._device:
                raise ValueError("seqlen tensor must be on the same device as q")
            if seqlen.dtype not in (torch.int32, torch.int64):
                raise ValueError("seqlen tensor must be int32 or int64")
            self._static_seqlen = seqlen
        else:
            seqlen_value = k_fp8.shape[1] if seqlen is None else int(seqlen)
            self._static_seqlen = torch.tensor(
                seqlen_value, device=self._device, dtype=torch.int32
            )

        self._static_q = torch.empty_like(q, device=self._device)
        if self._copy_kv:
            self._static_k_fp8 = torch.empty_like(k_fp8, device=self._device)
            self._static_v = torch.empty_like(v, device=self._device)
        else:
            if k_fp8.device != self._device or v.device != self._device:
                raise ValueError("k_fp8 and v must be on the same device as q")
            self._static_k_fp8 = k_fp8
            self._static_v = v
        self._static_k_residual = None
        if self._use_fp8_residual:
            if self._copy_kv:
                self._static_k_residual = torch.empty_like(k_residual, device=self._device)
            else:
                if k_residual.device != self._device:
                    raise ValueError("k_residual must be on the same device as q")
                self._static_k_residual = k_residual

        self._static_threshold = None
        if self._use_ext_th:
            self._static_threshold = torch.empty_like(precomputed_threshold, device=self._device)

        if workspace is None:
            B = q.shape[0]
            HQ = q.shape[2]
            HKV = v.shape[2]
            V = v.shape[3]
            T_full = k_fp8.shape[1]
            SBS = self._SBS if self._SBS is not None else self._BS
            ntb = (T_full + self._BS - 1) // self._BS
            nsb = (self._BS + SBS - 1) // SBS
            ntbs = ntb * nsb
            self._workspace = FP8FP8DecodeWorkspace(B, HQ, HKV, V, ntbs, self._device, q.dtype)
        else:
            B = q.shape[0]
            HQ = q.shape[2]
            HKV = v.shape[2]
            V = v.shape[3]
            T_full = k_fp8.shape[1]
            SBS = self._SBS if self._SBS is not None else self._BS
            ntb = (T_full + self._BS - 1) // self._BS
            nsb = (self._BS + SBS - 1) // SBS
            ntbs = ntb * nsb
            workspace.ensure(B, HQ, HKV, V, ntbs, self._device, q.dtype)
            self._workspace = workspace

        # Seed static buffers once to avoid uninitialized data in capture.
        self._static_q.copy_(q)
        if self._copy_kv:
            self._static_k_fp8.copy_(k_fp8)
            self._static_v.copy_(v)
        if self._use_fp8_residual:
            if self._copy_kv:
                self._static_k_residual.copy_(k_residual)
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        # Warmup to trigger Triton JIT before graph capture.
        for _ in range(max(1, warmup)):
            attn_forward_decode_fp8fp8(
                q=self._static_q,
                k_fp8=self._static_k_fp8,
                k_residual=self._static_k_residual,
                v=self._static_v,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                seqlen=self._static_seqlen,
                workspace=self._workspace,
            )
        torch.cuda.synchronize(self._device)

        self._graph = torch.cuda.CUDAGraph()
        self._pool = torch.cuda.graphs.graph_pool_handle()
        with torch.cuda.graph(self._graph, pool=self._pool):
            self._static_out = attn_forward_decode_fp8fp8(
                q=self._static_q,
                k_fp8=self._static_k_fp8,
                k_residual=self._static_k_residual,
                v=self._static_v,
                scale=self._scale,
                BS=self._BS,
                SBS=self._SBS,
                delta=self._delta,
                return_skip_ratio=False,
                precomputed_threshold=self._static_threshold,
                use_fp8_residual=self._use_fp8_residual,
                seqlen=self._static_seqlen,
                workspace=self._workspace,
            )

    @property
    def output(self) -> torch.Tensor:
        return self._static_out

    def replay(
        self,
        q: torch.Tensor,
        k_fp8: torch.Tensor,
        v: torch.Tensor,
        *,
        k_residual: Optional[torch.Tensor] = None,
        seqlen: int | torch.Tensor | None = None,
        precomputed_threshold: Optional[torch.Tensor] = None,
        return_skip_ratio: bool = False,
    ) -> torch.Tensor:
        if q.device != self._device:
            raise ValueError("q must be on the same device as the captured graph.")
        if self._use_fp8_residual and k_residual is None:
            raise ValueError("k_residual is required for this captured graph.")
        if self._use_ext_th and precomputed_threshold is None:
            raise ValueError("precomputed_threshold is required for this captured graph.")

        self._static_q.copy_(q)
        if self._copy_kv:
            self._static_k_fp8.copy_(k_fp8)
            self._static_v.copy_(v)
            if self._use_fp8_residual:
                self._static_k_residual.copy_(k_residual)
        else:
            if k_fp8 is not self._static_k_fp8 or v is not self._static_v:
                raise ValueError("k_fp8/v must match the captured tensors when copy_kv=False")
            if self._use_fp8_residual and k_residual is not self._static_k_residual:
                raise ValueError("k_residual must match the captured tensor when copy_kv=False")
        if seqlen is not None:
            if isinstance(seqlen, torch.Tensor):
                if seqlen.device != self._device:
                    raise ValueError("seqlen tensor must be on the same device as q")
                if seqlen.dtype not in (torch.int32, torch.int64):
                    raise ValueError("seqlen tensor must be int32 or int64")
                self._static_seqlen.copy_(seqlen)
            else:
                self._static_seqlen.fill_(int(seqlen))
        if self._use_ext_th:
            self._static_threshold.copy_(precomputed_threshold)

        self._graph.replay()
        if not return_skip_ratio:
            return self._static_out

        # NOTE: Skip ratio computation is not captured; it re-runs the kernel once.
        _, skip_ratio = attn_forward_decode_fp8fp8(
            q=self._static_q,
            k_fp8=self._static_k_fp8,
            k_residual=self._static_k_residual,
            v=self._static_v,
            scale=self._scale,
            BS=self._BS,
            SBS=self._SBS,
            delta=self._delta,
            return_skip_ratio=True,
            precomputed_threshold=self._static_threshold,
            use_fp8_residual=self._use_fp8_residual,
            seqlen=self._static_seqlen,
            workspace=self._workspace,
        )
        return self._static_out, skip_ratio

    __call__ = replay

    def replay_only(self) -> torch.Tensor:
        """Replay without updating static inputs."""
        self._graph.replay()
        return self._static_out
