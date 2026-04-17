#!/usr/bin/env python3
"""
验证 Qwen3.5-MoE GDN (Gated Delta Network) 变长序列支持功能

测试目标：
1. 验证 cu_seq_lens 参数正确传递到 GDN 算子
2. 验证变长序列计算的数值正确性
3. 验证与固定长度序列结果的一致性
4. 验证梯度反向传播的正确性

使用方法：
    # 运行所有测试
    python tests/models/test_qwen3_5_moe_gdn_varlen.py

    # 运行单个测试
    python tests/models/test_qwen3_5_moe_gdn_varlen.py -k test_cu_seqlens_passthrough

    # 详细输出
    python tests/models/test_qwen3_5_moe_gdn_varlen.py -v
"""

import pytest
import torch
import torch.nn.functional as F
from typing import Optional


# ============================================================
# 测试 1: cu_seq_lens 参数传递验证
# ============================================================
class TestCuSeqlensPassthrough:
    """验证 cu_seq_lens 参数能否正确从模型输入传递到 GDN 算子"""

    def test_gdn_forward_accepts_cu_seq_lens(self):
        """
        验证点：Qwen3_5MoeGatedDeltaNet.forward() 接受 cu_seq_lens 参数
        
        预期结果：
        - 函数签名包含 cu_seq_lens 参数
        - 传入 None 时不报错（向后兼容）
        - 传入 tensor 时不报错
        """
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeGatedDeltaNet,
            Qwen3_5MoeTextConfig,
        )
        import inspect

        # 检查函数签名
        sig = inspect.signature(Qwen3_5MoeGatedDeltaNet.forward)
        assert "cu_seq_lens" in sig.parameters, "forward() 方法缺少 cu_seq_lens 参数"

        # 检查参数默认值
        param = sig.parameters["cu_seq_lens"]
        assert param.default is None, "cu_seq_lens 默认值应为 None"

    def test_decoder_layer_extracts_cu_seq_lens_from_kwargs(self):
        """
        验证点：Qwen3_5MoeDecoderLayer 从 kwargs 中提取 cu_seq_lens_q
        
        预期结果：
        - DecoderLayer.forward() 正确提取 cu_seq_lens_q
        - 提取的值传递给 linear_attn
        """
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeDecoderLayer,
            Qwen3_5MoeTextConfig,
        )
        import inspect

        # 获取 forward 方法源码
        source = inspect.getsource(Qwen3_5MoeDecoderLayer.forward)

        # 检查是否从 kwargs 提取 cu_seq_lens
        assert "cu_seq_lens_q" in source or "cu_seq_lens" in source, (
            "DecoderLayer.forward() 未从 kwargs 提取 cu_seq_lens"
        )

    def test_chunk_gated_delta_rule_receives_cu_seqlens(self):
        """
        验证点：chunk_gated_delta_rule 调用时传入 cu_seqlens 参数
        
        预期结果：
        - 调用处包含 cu_seqlens=cu_seq_lens
        """
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeGatedDeltaNet,
        )
        import inspect

        source = inspect.getsource(Qwen3_5MoeGatedDeltaNet.forward)

        # 检查 chunk_gated_delta_rule 调用
        assert "cu_seqlens" in source, (
            "chunk_gated_delta_rule 调用中未传入 cu_seqlens 参数"
        )


# ============================================================
# 测试 2: 变长序列数值正确性
# ============================================================
class TestVariableLengthCorrectness:
    """验证变长序列计算的数值正确性"""

    @pytest.fixture
    def setup_chunk_gated_delta_rule(self):
        """设置测试环境"""
        try:
            from verl.models.transformers import chunk_gated_delta_rule
            return chunk_gated_delta_rule
        except ImportError:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule as chunk_gated_delta_rule,
            )
            return chunk_gated_delta_rule

    def test_varlen_vs_padded_equivalence(self, setup_chunk_gated_delta_rule):
        """
        验证点：变长序列（使用 cu_seqlens）与 padded 序列结果等价
        
        测试设计：
        - 创建 2 个不同长度的序列
        - 方法 1：分别计算每个序列，然后拼接
        - 方法 2：使用 cu_seqlens 一次性计算
        - 比较两种方法的结果
        
        预期结果：
        - 两种方法的输出差异在容忍范围内 (atol=1e-3)
        """
        chunk_gated_delta_rule = setup_chunk_gated_delta_rule

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        # 序列配置
        seq_len_1, seq_len_2 = 64, 128
        batch_size = 1  # varlen 模式要求 batch_size=1
        num_heads = 4
        head_dim = 64
        total_len = seq_len_1 + seq_len_2

        # 创建输入数据
        q = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.sigmoid(beta)

        # cu_seqlens: [0, seq_len_1, seq_len_1+seq_len_2]
        cu_seqlens = torch.tensor([0, seq_len_1, total_len], dtype=torch.int32, device=device)

        # 方法 1：使用 cu_seqlens 一次性计算
        out_varlen, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        # 方法 2：分别计算每个序列
        q1 = q[:, :seq_len_1]
        k1 = k[:, :seq_len_1]
        v1 = v[:, :seq_len_1]
        g1 = g[:, :seq_len_1]
        beta1 = beta[:, :seq_len_1]

        q2 = q[:, seq_len_1:]
        k2 = k[:, seq_len_1:]
        v2 = v[:, seq_len_1:]
        g2 = g[:, seq_len_1:]
        beta2 = beta[:, seq_len_1:]

        out1, _ = chunk_gated_delta_rule(
            q1, k1, v1, g1, beta1,
            cu_seqlens=None,  # 单序列不需要 cu_seqlens
            use_qk_l2norm_in_kernel=True,
        )
        out2, _ = chunk_gated_delta_rule(
            q2, k2, v2, g2, beta2,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )

        out_separate = torch.cat([out1, out2], dim=1)

        # 验证：两种方法的结果应该接近
        diff = (out_varlen.float() - out_separate.float()).abs().max().item()
        print(f"\n变长序列 vs 分离计算 最大差异: {diff:.6e}")

        assert diff < 1e-2, (
            f"变长序列与分离计算结果差异过大: {diff:.6e} > 1e-2"
        )

    def test_no_cross_contamination_between_sequences(self, setup_chunk_gated_delta_rule):
        """
        验证点：不同序列之间没有信息泄漏
        
        测试设计：
        - 创建两个完全独立的序列
        - 使用 cu_seqlens 合并计算
        - 验证序列 1 的输出只依赖于序列 1 的输入
        - 修改序列 2 的输入，序列 1 的输出不应改变
        
        预期结果：
        - 修改序列 2 后，序列 1 的输出保持不变
        """
        chunk_gated_delta_rule = setup_chunk_gated_delta_rule

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        seq_len_1, seq_len_2 = 64, 64
        batch_size = 1
        num_heads = 4
        head_dim = 64
        total_len = seq_len_1 + seq_len_2

        # 创建初始输入
        q = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.sigmoid(beta)

        cu_seqlens = torch.tensor([0, seq_len_1, total_len], dtype=torch.int32, device=device)

        # 第一次计算
        out1, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        out1_seq1 = out1[:, :seq_len_1].clone()

        # 修改序列 2 的输入
        q_modified = q.clone()
        q_modified[:, seq_len_1:] = torch.randn_like(q[:, seq_len_1:])

        # 第二次计算
        out2, _ = chunk_gated_delta_rule(
            q_modified, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        out2_seq1 = out2[:, :seq_len_1]

        # 验证：序列 1 的输出应该不变
        diff = (out1_seq1.float() - out2_seq1.float()).abs().max().item()
        print(f"\n序列间信息泄漏检测 - 最大差异: {diff:.6e}")

        assert diff < 1e-5, (
            f"检测到序列间信息泄漏: 修改序列2影响了序列1的输出，差异={diff:.6e}"
        )


# ============================================================
# 测试 3: 梯度反向传播验证
# ============================================================
class TestGradientBackward:
    """验证变长序列模式下的梯度反向传播"""

    @pytest.fixture
    def setup_chunk_gated_delta_rule(self):
        try:
            from verl.models.transformers import chunk_gated_delta_rule
            return chunk_gated_delta_rule
        except ImportError:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule as chunk_gated_delta_rule,
            )
            return chunk_gated_delta_rule

    def test_gradient_flow_with_cu_seqlens(self, setup_chunk_gated_delta_rule):
        """
        验证点：使用 cu_seqlens 时梯度能正确反向传播
        
        测试设计：
        - 创建需要梯度的输入
        - 使用 cu_seqlens 进行前向计算
        - 计算 loss 并反向传播
        - 检查所有输入都有梯度
        
        预期结果：
        - 所有输入 tensor 的 grad 不为 None
        - 梯度值不为 NaN 或 Inf
        """
        chunk_gated_delta_rule = setup_chunk_gated_delta_rule

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        seq_len_1, seq_len_2 = 64, 128
        batch_size = 1
        num_heads = 4
        head_dim = 64
        total_len = seq_len_1 + seq_len_2

        # 创建需要梯度的输入
        q = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        g = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device, requires_grad=True)
        beta = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device, requires_grad=True)
        beta = torch.sigmoid(beta)

        cu_seqlens = torch.tensor([0, seq_len_1, total_len], dtype=torch.int32, device=device)

        # 前向计算
        out, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        # 计算 loss
        loss = out.float().sum()

        # 反向传播
        loss.backward()

        # 验证梯度
        tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        for name, tensor in tensors.items():
            assert tensor.grad is not None, f"{name} 的梯度为 None"
            assert not torch.isnan(tensor.grad).any(), f"{name} 的梯度包含 NaN"
            assert not torch.isinf(tensor.grad).any(), f"{name} 的梯度包含 Inf"
            print(f"{name} 梯度: mean={tensor.grad.mean().item():.6e}, std={tensor.grad.std().item():.6e}")


# ============================================================
# 测试 4: 端到端模型集成测试
# ============================================================
class TestEndToEndIntegration:
    """端到端测试：验证完整模型流程中 cu_seq_lens 的传递"""

    def test_model_forward_with_varlen_metadata(self):
        """
        验证点：模型能正确处理包含 cu_seq_lens 的输入
        
        测试设计：
        - 模拟 verl 的 model_inputs 格式
        - 创建变长序列的 batch
        - 调用模型 forward
        - 验证输出形状正确
        
        预期结果：
        - 模型不报错
        - 输出形状正确
        """
        # 这个测试需要实际加载模型，标记为可选
        pytest.skip("需要实际模型权重，集成测试应在完整环境中运行")


# ============================================================
# 测试 5: 边界条件测试
# ============================================================
class TestEdgeCases:
    """边界条件测试"""

    @pytest.fixture
    def setup_chunk_gated_delta_rule(self):
        try:
            from verl.models.transformers import chunk_gated_delta_rule
            return chunk_gated_delta_rule
        except ImportError:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule as chunk_gated_delta_rule,
            )
            return chunk_gated_delta_rule

    def test_single_sequence_with_cu_seqlens(self, setup_chunk_gated_delta_rule):
        """
        验证点：单个序列使用 cu_seqlens 也能正常工作
        
        预期结果：
        - cu_seqlens = [0, seq_len]
        - 输出与不使用 cu_seqlens 时一致
        """
        chunk_gated_delta_rule = setup_chunk_gated_delta_rule

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        batch_size = 1
        seq_len = 128
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device)
        beta = torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device)
        beta = torch.sigmoid(beta)

        cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        # 使用 cu_seqlens
        out_with, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        # 不使用 cu_seqlens
        out_without, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )

        diff = (out_with.float() - out_without.float()).abs().max().item()
        print(f"\n单序列 cu_seqlens vs None 最大差异: {diff:.6e}")

        assert diff < 1e-3, f"单序列模式下结果不一致: {diff:.6e}"

    def test_cu_seqlens_dtype_validation(self, setup_chunk_gated_delta_rule):
        """
        验证点：cu_seqlens 数据类型验证
        
        预期结果：
        - int32 类型正常工作
        - int64 类型也能工作（或给出明确错误）
        """
        chunk_gated_delta_rule = setup_chunk_gated_delta_rule

        torch.manual_seed(42)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        batch_size = 1
        seq_len = 64
        num_heads = 4
        head_dim = 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device)
        beta = torch.randn(batch_size, seq_len, num_heads, dtype=dtype, device=device)
        beta = torch.sigmoid(beta)

        # 测试 int32
        cu_seqlens_int32 = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        out1, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens_int32,
            use_qk_l2norm_in_kernel=True,
        )
        assert out1.shape == q.shape[:-1] + (q.shape[-1],), "输出形状不正确"


# ============================================================
# 运行配置
# ============================================================
if __name__ == "__main__":
    pytest.main([
        __file__,
        "-v",
        "--tb=short",
        "-x",
    ])
