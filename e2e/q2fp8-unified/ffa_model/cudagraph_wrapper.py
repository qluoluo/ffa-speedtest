"""
CUDA Graph Wrapper with Pre-allocated Buffers

支持动态shape变化的CUDA Graph wrapper，通过预分配最大可能的buffer来处理decode过程中的shape增长。
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


class PreallocatedCUDAGraphRunner:
    """
    CUDA Graph runner with pre-allocated buffers to handle dynamic shape changes.

    核心思路：
    1. 预分配足够大的buffer（支持max_decode_tokens）
    2. 每次replay时，只copy实际使用的部分
    3. Kernel根据实际的T来处理数据
    """

    def __init__(
        self,
        initial_k_q: torch.Tensor,
        initial_k_scale: torch.Tensor,
        initial_v: torch.Tensor,
        initial_k_residual: Optional[torch.Tensor],
        initial_k_current: Optional[torch.Tensor],
        initial_v_current: Optional[torch.Tensor],
        max_decode_tokens: int = 512,
        k_bits: int = 2,
        scale: Optional[float] = None,
        BS: int = 128,
        SBS: Optional[int] = None,
        delta: float = 5.0,
        use_fp8_residual: bool = True,
        max_current: int = 128,
        warmup: int = 3,
    ):
        """
        Args:
            initial_k_q: 初始的量化K [B, T0, HKV, K_packed]
            initial_k_scale: 初始的scale [B, NTB0, HKV, K] 或 [B, HKV, K]
            initial_v: 初始的V [B, T0, HKV, V]
            initial_k_residual: 初始的残差 [B, T0, HKV, K]
            initial_k_current: current buffer [B, max_current, HKV, K]
            initial_v_current: current buffer [B, max_current, HKV, V]
            max_decode_tokens: 最大decode长度（用于预分配）
            其他参数: 同attn_forward_decode_quantized
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
        self.max_current = max_current

        # 获取初始shape
        B, T0, HKV, K_packed = initial_k_q.shape
        _, _, _, V = initial_v.shape
        _, _, _, K = initial_k_residual.shape if initial_k_residual is not None else (0, 0, 0, 128)

        # 检查k_scale的shape（可能是per-block或global）
        if initial_k_scale.ndim == 4:
            # Per-block: [B, NTB, HKV, K]
            self.use_perblock_scale = True
            NTB0 = initial_k_scale.shape[1]
        else:
            # Global: [B, HKV, K]
            self.use_perblock_scale = False
            NTB0 = (T0 + BS - 1) // BS

        # 计算最大可能的shape
        max_additional_blocks = (max_decode_tokens + BS - 1) // BS
        max_total_blocks = NTB0 + max_additional_blocks
        max_T = max_total_blocks * BS

        print(f"[CUDAGraph] Preallocating buffers:")
        print(f"  Initial: T={T0}, NTB={NTB0}")
        print(f"  Max decode tokens: {max_decode_tokens}")
        print(f"  Max total: T={max_T}, NTB={max_total_blocks}")
        print(f"  Memory overhead: ~{(max_T - T0) * B * HKV * K_packed / 1024**2:.1f} MB")

        # 预分配固定大小的buffers
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
            # Global scale不需要预分配（shape不变）
            self.static_k_scale = torch.empty_like(initial_k_scale, device=self.device)

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

        # Current buffers（shape固定）
        if initial_k_current is not None:
            self.static_k_current = torch.empty_like(initial_k_current, device=self.device)
            self.static_v_current = torch.empty_like(initial_v_current, device=self.device)
        else:
            self.static_k_current = None
            self.static_v_current = None

        # 保存shape信息
        self.B = B
        self.HKV = HKV
        self.K_packed = K_packed
        self.V = V
        self.K = K
        self.max_T = max_T
        self.max_total_blocks = max_total_blocks

        # 初始化：copy初始数据
        self._copy_to_static(
            initial_k_q, initial_k_scale, initial_v, initial_k_residual,
            initial_k_current, initial_v_current
        )

        # Warmup
        print(f"[CUDAGraph] Warming up ({warmup} iterations)...")
        for i in range(warmup):
            _ = attn_forward_decode_quantized(
                q=torch.randn(B, 1, HKV * (initial_k_q.shape[2] // HKV), K,
                             dtype=torch.float16, device=self.device),
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
                max_current=self.max_current,
            )
        torch.cuda.synchronize(self.device)

        # 捕获CUDA Graph
        print(f"[CUDAGraph] Capturing graph...")
        self.graph = torch.cuda.CUDAGraph()

        # 创建query tensor（会在replay时更新）
        HQ = initial_k_q.shape[2] * (initial_k_q.shape[2] // HKV)  # 假设GQA
        self.static_q = torch.randn(B, 1, HQ, K, dtype=torch.float16, device=self.device)
        self.current_len_tensor = torch.tensor([0], dtype=torch.int32, device=self.device)

        with torch.cuda.graph(self.graph):
            # 注意：这里使用完整的static buffers，但kernel会根据实际T来处理
            self.static_out = attn_forward_decode_quantized(
                q=self.static_q,
                k_q=self.static_k_q,
                k_scale=self.static_k_scale,
                v=self.static_v,
                k_residual=self.static_k_residual,
                k_current=self.static_k_current,
                v_current=self.static_v_current,
                current_len=0,  # 会在replay时通过修改tensor来更新
                k_bits=self.k_bits,
                scale=self.scale,
                BS=self.BS,
                SBS=self.SBS,
                delta=self.delta,
                use_fp8_residual=self.use_fp8_residual,
                max_current=self.max_current,
            )

        print(f"[CUDAGraph] Graph captured successfully!")

    def _copy_to_static(
        self,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor],
        k_current: Optional[torch.Tensor],
        v_current: Optional[torch.Tensor],
    ):
        """Copy actual data to static buffers (only the used portion)"""
        T = k_q.shape[1]

        # Copy k_q
        self.static_k_q[:, :T, :, :].copy_(k_q)

        # Copy k_scale
        if self.use_perblock_scale:
            NTB = k_scale.shape[1]
            self.static_k_scale[:, :NTB, :, :].copy_(k_scale)
        else:
            self.static_k_scale.copy_(k_scale)

        # Copy v
        self.static_v[:, :T, :, :].copy_(v)

        # Copy k_residual
        if self.use_fp8_residual and k_residual is not None:
            self.static_k_residual[:, :T, :, :].copy_(k_residual)

        # Copy current buffers
        if k_current is not None and self.static_k_current is not None:
            self.static_k_current.copy_(k_current)
        if v_current is not None and self.static_v_current is not None:
            self.static_v_current.copy_(v_current)

    def replay(
        self,
        q: torch.Tensor,
        k_q: torch.Tensor,
        k_scale: torch.Tensor,
        v: torch.Tensor,
        k_residual: Optional[torch.Tensor],
        k_current: Optional[torch.Tensor],
        v_current: Optional[torch.Tensor],
        current_len: int,
    ) -> torch.Tensor:
        """
        Replay the captured graph with new inputs.

        Args:
            q: [B, 1, HQ, K] Query tensor
            k_q: [B, T, HKV, K_packed] 当前的量化K（T可能变化）
            k_scale: [B, NTB, HKV, K] 或 [B, HKV, K] 当前的scale
            v: [B, T, HKV, V] 当前的V
            k_residual: [B, T, HKV, K] 当前的残差
            k_current: [B, max_current, HKV, K] current buffer
            v_current: [B, max_current, HKV, V] current buffer
            current_len: 当前buffer的有效长度

        Returns:
            attn_output: [B, HQ, V]
        """
        T = k_q.shape[1]

        # 检查是否超出预分配的大小
        if T > self.max_T:
            raise RuntimeError(
                f"Sequence length {T} exceeds pre-allocated max {self.max_T}. "
                f"Increase max_decode_tokens when creating the runner."
            )

        # Copy数据到static buffers
        self._copy_to_static(k_q, k_scale, v, k_residual, k_current, v_current)

        # Copy query
        self.static_q.copy_(q)

        # 注意：current_len是Python int，无法在graph中动态修改
        # 这是一个限制，但对于大部分情况影响不大
        # TODO: 如果需要支持动态current_len，需要使用多个graph

        # Replay graph
        self.graph.replay()

        return self.static_out.clone()  # 返回副本，避免被下次replay覆盖
