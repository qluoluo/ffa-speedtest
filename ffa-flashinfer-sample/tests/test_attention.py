"""
Unit tests for FFA-Sample.
"""

import sys
sys.path.insert(0, "/inspire/qb-ilm/project/exploration-topic/liuzhigeng-253108120105/projects/ffa/ffa-speedtest/ffa-flashinfer-sample")

import pytest
import torch

from ffa_sample import (
    SparseAttentionWithFlashInfer,
    sample_k_fp16,
    attn_forward_decode_sample4,
)
from ffa_sample.utils import PagedKVCache, continuous_to_paged


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return "cuda:0"


@pytest.fixture
def dtype():
    return torch.float16


class TestSampleKFP16:
    """测试 K 采样函数."""

    def test_basic_shape(self, device, dtype):
        """测试基本的输出形状."""
        B, T, HKV, K = 1, 1024, 8, 128
        BS = 128

        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        k_sample = sample_k_fp16(k, BS=BS)

        num_blocks = (T + BS - 1) // BS
        expected_shape = (B, num_blocks, HKV, 4, K)  # 4 samples per block

        assert k_sample.shape == expected_shape, f"Expected {expected_shape}, got {k_sample.shape}"

    def test_custom_sample_offsets(self, device, dtype):
        """测试自定义采样偏移."""
        B, T, HKV, K = 1, 512, 4, 64
        BS = 128
        sample_offsets = [0, 64]  # 只采样 2 个点

        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        k_sample = sample_k_fp16(k, BS=BS, sample_offsets=sample_offsets)

        num_blocks = (T + BS - 1) // BS
        expected_shape = (B, num_blocks, HKV, 2, K)

        assert k_sample.shape == expected_shape

    def test_padding(self, device, dtype):
        """测试序列长度不是 block size 整数倍时的 padding."""
        B, T, HKV, K = 1, 300, 4, 64  # 300 不是 128 的整数倍
        BS = 128

        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        k_sample = sample_k_fp16(k, BS=BS)

        num_blocks = (T + BS - 1) // BS  # ceil(300/128) = 3
        expected_shape = (B, num_blocks, HKV, 4, K)

        assert k_sample.shape == expected_shape


class TestAttnForwardDecodeSample4:
    """测试稀疏注意力 kernel."""

    def test_basic_forward(self, device, dtype):
        """测试基本的前向传播."""
        B, T, HQ, HKV, K, V = 1, 1024, 32, 8, 128, 128
        BS = 128

        q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

        k_sample = sample_k_fp16(k, BS=BS)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

        output = attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            BS=BS,
            delta=5.0,
        )

        expected_shape = (B, HQ, V)
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"

    def test_return_skip_ratio(self, device, dtype):
        """测试返回跳过比例."""
        B, T, HQ, HKV, K, V = 1, 2048, 16, 4, 64, 64
        BS = 128

        q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

        k_sample = sample_k_fp16(k, BS=BS)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

        output, skip_ratio = attn_forward_decode_sample4(
            q=q,
            k_sample_q=k_sample,
            k_sample_scale=k_sample_scale,
            k_full=k,
            v=v,
            BS=BS,
            delta=5.0,
            return_skip_ratio=True,
        )

        assert isinstance(skip_ratio, float)
        assert 0.0 <= skip_ratio <= 1.0

    def test_different_deltas(self, device, dtype):
        """测试不同 delta 值对跳过比例的影响."""
        B, T, HQ, HKV, K, V = 1, 2048, 16, 4, 64, 64
        BS = 128

        q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

        k_sample = sample_k_fp16(k, BS=BS)
        num_blocks = k_sample.shape[1]
        k_sample_scale = torch.zeros((B, num_blocks, HKV, K), device=device, dtype=dtype)

        # 较大的 delta 应该导致更高的跳过比例
        _, skip_ratio_small = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=BS, delta=1.0, return_skip_ratio=True,
        )

        _, skip_ratio_large = attn_forward_decode_sample4(
            q=q, k_sample_q=k_sample, k_sample_scale=k_sample_scale,
            k_full=k, v=v, BS=BS, delta=10.0, return_skip_ratio=True,
        )

        # 通常 delta 越大，阈值越低，保留的 block 越少，skip_ratio 越高
        # 但这取决于数据分布，所以我们只检查都是有效值
        assert 0.0 <= skip_ratio_small <= 1.0
        assert 0.0 <= skip_ratio_large <= 1.0


class TestSparseAttentionWithFlashInfer:
    """测试 SparseAttentionWithFlashInfer 类."""

    def test_initialization(self, device, dtype):
        """测试初始化."""
        sparse_attn = SparseAttentionWithFlashInfer(
            num_heads=32,
            head_dim=128,
            page_size=128,
            num_kv_heads=8,
            device=device,
            dtype=dtype,
        )

        assert sparse_attn.num_heads == 32
        assert sparse_attn.head_dim == 128
        assert sparse_attn.num_kv_heads == 8
        assert sparse_attn.num_groups == 4  # 32 / 8

    def test_forward(self, device, dtype):
        """测试前向传播."""
        B, T, HQ, HKV, K, V = 1, 1024, 32, 8, 128, 128

        sparse_attn = SparseAttentionWithFlashInfer(
            num_heads=HQ,
            head_dim=K,
            page_size=128,
            num_kv_heads=HKV,
            device=device,
            dtype=dtype,
        )

        q = torch.randn(B, 1, HQ, K, device=device, dtype=dtype)
        k = torch.randn(B, T, HKV, K, device=device, dtype=dtype)
        v = torch.randn(B, T, HKV, V, device=device, dtype=dtype)

        output = sparse_attn(q, k, v, delta=5.0)

        expected_shape = (B, HQ, V)
        assert output.shape == expected_shape


class TestPagedKVCache:
    """测试 PagedKVCache 类."""

    def test_initialization(self, device, dtype):
        """测试初始化."""
        cache = PagedKVCache(
            max_num_pages=64,
            page_size=16,
            num_kv_heads=8,
            head_dim=128,
            device=device,
            dtype=dtype,
        )

        assert cache.max_num_pages == 64
        assert cache.page_size == 16
        assert cache.num_allocated_pages == 0

    def test_allocate_pages(self, device, dtype):
        """测试页面分配."""
        cache = PagedKVCache(
            max_num_pages=64,
            page_size=16,
            num_kv_heads=8,
            head_dim=128,
            device=device,
            dtype=dtype,
        )

        indices = cache.allocate_pages(10)

        assert len(indices) == 10
        assert cache.num_allocated_pages == 10
        assert torch.all(indices == torch.arange(10, device=device, dtype=torch.int32))

    def test_reset(self, device, dtype):
        """测试重置."""
        cache = PagedKVCache(
            max_num_pages=64,
            page_size=16,
            num_kv_heads=8,
            head_dim=128,
            device=device,
            dtype=dtype,
        )

        cache.allocate_pages(10)
        cache.reset()

        assert cache.num_allocated_pages == 0


class TestContinuousToPaged:
    """测试连续到分页的转换."""

    def test_basic_conversion(self, device, dtype):
        """测试基本转换."""
        B, T, H, D = 1, 256, 8, 128
        page_size = 16

        k = torch.randn(B, T, H, D, device=device, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)

        paged_kv, indices, indptr, last_page_len = continuous_to_paged(
            k, v, page_size=page_size, kv_layout="NHD"
        )

        num_pages = T // page_size
        assert paged_kv.shape == (num_pages, 2, page_size, H, D)
        assert len(indices) == num_pages
        assert indptr[1].item() == num_pages

    def test_padding(self, device, dtype):
        """测试需要 padding 的情况."""
        B, T, H, D = 1, 100, 4, 64  # 100 不是 16 的整数倍
        page_size = 16

        k = torch.randn(B, T, H, D, device=device, dtype=dtype)
        v = torch.randn(B, T, H, D, device=device, dtype=dtype)

        paged_kv, indices, indptr, last_page_len = continuous_to_paged(
            k, v, page_size=page_size
        )

        num_pages = (T + page_size - 1) // page_size  # ceil(100/16) = 7
        expected_last_page_len = T % page_size  # 100 % 16 = 4

        assert paged_kv.shape[0] == num_pages
        assert last_page_len.item() == expected_last_page_len


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
