"""
Simple CUDA Graph Wrapper for Q2FP8 Decode

简化版本：禁用current buffer，避免current_len的动态变化问题。
这样可以100%使用CUDA Graph，代价是所有tokens立即量化。
"""

import torch
import sys
from pathlib import Path
from typing import Optional

# Add attn_kernel path
_KERNEL_PATH = Path(__file__).parent.parent / "attn_kernel"
if str(_KERNEL_PATH) not in sys.path:
    sys.path.insert(0, str(_KERNEL_PATH))

from attn_q2fp8_unified import attn_forward_decode_quantized


class SimpleCUDAGraphRunner:
    """
    简化的CUDA Graph runner，通过禁用current buffer来避免动态变化。

    策略：
    1. max_current = 1（几乎禁用current buffer）
    2. current_len只有0或1两种状态
    3. 预分配buffer处理shape增长
    4. 为current_len=0和1各创建一个graph
    """

    def __init__(
        self,
        initial_k_q: torch.Tensor,
        initial_k_scale: torch.Tensor,
        initial_v: torch.Tensor,
        initial_k_residual: Optional[torch.Tensor],
        max_decode_tokens: int = 512,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        use_fp8_residual: bool = True,
        warmup: int = 2,
    ):
        """
        Args:
            initial_k_q: 初始的量化K [B, T0, HKV, K_packed]
            initial_k_scale: 初始的scale [B, NTB0, HKV, K] 或 [B, HKV, K]
            initial_v: 初始的V [B, T0, HKV, V]
            initial_k_residual: 初始的残差 [B, T0, HKV, K]
            max_decode_tokens: 最大decode长度
        """
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDAGraph")

        self.device = initial_k_q.device
        self.k_bits = k_bits
        self.scale = scale
        self.BS = BS
        self.SBS = SBS
        self.delta = delta
        self.use_fp8_residual = use_fp8_residual
        self.max_current = 1  # 固定为1

        # 获取shape
        B, T0, HKV, K_packed = initial_k_q.shape
        _, _, _, V = initial_v.shape
        K = initial_k_residual.shape[3] if initial_k_residual is not None else 128

        # 检查k_scale类型
        if initial_k_scale.ndim == 4:
            self.use_perblock_scale = True
            NTB0 = initial_k_scale.shape[1]
        else:
            self.use_perblock_scale = False
            NTB0 = (T0 + BS - 1) // BS

        # 计算最大shape
        max_additional_blocks = (max_decode_tokens + BS - 1) // BS
        max_total_blocks = NTB0 + max_additional_blocks
        max_T = max_total_blocks * BS

        print(f"\n{'='*70}")
        print(f"[SimpleCUDAGraph] Initializing")
        print(f"{'='*70}")
        print(f"Initial state:")
        print(f"  T={T0}, NTB={NTB0}")
        print(f"  Shape: k_q={list(initial_k_q.shape)}")
        print(f"\nMax capacity:")
        print(f"  max_decode_tokens={max_decode_tokens}")
        print(f"  max_T={max_T}, max_NTB={max_total_blocks}")
        print(f"\nMemory overhead:")
        overhead_mb = (max_T - T0) * B * HKV * K_packed / 1024**2
        print(f"  k_q: ~{overhead_mb:.1f} MB")
        print(f"  Total: ~{overhead_mb * 3:.1f} MB (k_q + v + k_residual)")
        print(f"{'='*70}\n")

        # 预分配buffers
        self.static_k_q = torch.zeros(
            (B, max_T, HKV, K_packed),
            dtype=initial_k_q.dtype,
            device=self.device
        )

        if self.use_perblock_scale:
            self.static_k_scale = torch.zeros(
                (B, max_total_blocks, HKV, K),
                dtype=initial_k_scale.dtype,
                device=self.device
            )
        else:
            self.static_k_scale = torch.empty_like(initial_k_scale)

        self.static_v = torch.zeros(
            (B, max_T, HKV, V),
            dtype=initial_v.dtype,
            device=self.device
        )

        if self.use_fp8_residual:
            self.static_k_residual = torch.zeros(
                (B, max_T, HKV, K),
                dtype=initial_k_residual.dtype,
                device=self.device
            )
        else:
            self.static_k_residual = None

        # Current buffers (固定大小=1)
        self.static_k_current = torch.zeros(
            (B, 1, HKV, K),
            dtype=torch.float16,
            device=self.device
        )
        self.static_v_current = torch.zeros(
            (B, 1, HKV, V),
            dtype=torch.float16,
            device=self.device
        )

        # 保存shape
        self.B = B
        self.HKV = HKV
        self.HQ = HKV  # 假设没有GQA，或者会在replay时更新
        self.K = K
        self.V = V
        self.K_packed = K_packed
        self.max_T = max_T

        # 初始化static buffers
        self._copy_to_static(initial_k_q, initial_k_scale, initial_v, initial_k_residual)

        # 创建query tensor
        self.static_q = torch.zeros(B, 1, HKV, K, dtype=torch.float16, device=self.device)

        # Warmup
        print(f"Warming up...")
        for _ in range(warmup):
            _ = attn_forward_decode_quantized(
                q=self.static_q,
                k_q=self.static_k_q[:, :T0, :, :],
                k_scale=self.static_k_scale if not self.use_perblock_scale else self.static_k_scale[:, :NTB0, :, :],
                v=self.static_v[:, :T0, :, :],
                k_residual=self.static_k_residual[:, :T0, :, :] if self.use_fp8_residual else None,
                k_current=self.static_k_current,
                v_current=self.static_v_current,
                current_len=0,
                k_bits=self.k_bits,
                scale=self.scale,
                BS=self.BS,
                SBS=self.SBS,
                delta=self.delta,
                use_fp8_residual=self.use_fp8_residual,
                max_current=1,
            )
        torch.cuda.synchronize()

        # 为current_len=0和1各创建一个graph
        print(f"Capturing CUDA Graphs...")
        self.graphs = {}

        for cl in [0, 1]:
            print(f"  Capturing graph for current_len={cl}...")
            graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(graph):
                out = attn_forward_decode_quantized(
                    q=self.static_q,
                    k_q=self.static_k_q,
                    k_scale=self.static_k_scale,
                    v=self.static_v,
                    k_residual=self.static_k_residual,
                    k_current=self.static_k_current,
                    v_current=self.static_v_current,
                    current_len=cl,
                    k_bits=self.k_bits,
                    scale=self.scale,
                    BS=self.BS,
                    SBS=self.SBS,
                    delta=self.delta,
                    use_fp8_residual=self.use_fp8_residual,
                    max_current=1,
                )

            self.graphs[cl] = (graph, out)

        print(f"CUDA Graphs captured successfully!\n")

    def _copy_to_static(
        self,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor],
    ):
        """Copy data to static buffers"""
        T = k_q.shape[1]
        self.static_k_q[:, :T, :, :].copy_(k_q)

        if self.use_perblock_scale:
            NTB = k_scale.shape[1]
            self.static_k_scale[:, :NTB, :, :].copy_(k_scale)
        else:
            self.static_k_scale.copy_(k_scale)

        self.static_v[:, :T, :, :].copy_(v)

        if self.use_fp8_residual and k_residual is not None:
            self.static_k_residual[:, :T, :, :].copy_(k_residual)

    def replay(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor],
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        current_len: int,
    ) -> torch.Tensor:
        """Replay graph with new inputs"""
        T = k_q.shape[1]

        if T > self.max_T:
            raise RuntimeError(f"T={T} exceeds max_T={self.max_T}")

        if current_len not in [0, 1]:
            raise ValueError(f"current_len must be 0 or 1, got {current_len}")

        # Copy data
        self._copy_to_static(k_q, k_scale, v, k_residual)
        self.static_q.copy_(q)

        if current_len == 1:
            self.static_k_current[:, 0, :, :].copy_(k_current[:, 0, :, :])
            self.static_v_current[:, 0, :, :].copy_(v_current[:, 0, :, :])

        # Replay
        graph, out = self.graphs[current_len]
        graph.replay()

        return out.clone()
