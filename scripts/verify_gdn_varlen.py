#!/usr/bin/env python3
"""
快速验证脚本：检查 GDN 变长序列支持是否正确实施

运行方式：
    python scripts/verify_gdn_varlen.py

输出：
    - 代码修改状态检查
    - 参数传递链路验证
    - 基础功能测试
"""

import sys
import torch
from pathlib import Path


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def check_code_modifications():
    """检查代码修改是否正确实施"""
    print_section("步骤 1: 检查代码修改状态")

    # 查找文件
    import os
    import glob

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if not conda_prefix:
        print("ERROR: 未找到 CONDA_PREFIX 环境变量")
        return False

    matches = glob.glob(
        os.path.join(
            conda_prefix,
            "lib",
            "python*",
            "site-packages",
            "transformers",
            "models",
            "qwen3_5_moe",
            "modeling_qwen3_5_moe.py",
        )
    )

    if not matches:
        print("ERROR: 找不到 modeling_qwen3_5_moe.py")
        return False

    file_path = Path(matches[0])
    print(f"找到文件: {file_path}")

    with open(file_path, "r") as f:
        content = f.read()

    checks = [
        (
            "Qwen3_5MoeGatedDeltaNet.forward 包含 cu_seq_lens 参数",
            "cu_seq_lens: torch.Tensor | None = None" in content,
        ),
        (
            "chunk_gated_delta_rule 调用传入 cu_seqlens",
            "cu_seqlens=cu_seq_lens" in content,
        ),
        (
            "DecoderLayer 从 kwargs 提取 cu_seq_lens",
            'kwargs.get("cu_seq_lens_q"' in content or "cu_seq_lens_q" in content,
        ),
    ]

    all_passed = True
    for desc, passed in checks:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {desc}")
        if not passed:
            all_passed = False

    return all_passed


def test_cu_seqlens_passthrough():
    """测试 cu_seq_lens 参数传递"""
    print_section("步骤 2: 测试参数传递链路")

    try:
        import inspect
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
            Qwen3_5MoeGatedDeltaNet,
            Qwen3_5MoeDecoderLayer,
        )

        # 检查 GDN forward 签名
        sig = inspect.signature(Qwen3_5MoeGatedDeltaNet.forward)
        if "cu_seq_lens" in sig.parameters:
            print("  [PASS] Qwen3_5MoeGatedDeltaNet.forward 接受 cu_seq_lens 参数")
        else:
            print("  [FAIL] Qwen3_5MoeGatedDeltaNet.forward 缺少 cu_seq_lens 参数")
            return False

        # 检查 DecoderLayer forward 源码
        source = inspect.getsource(Qwen3_5MoeDecoderLayer.forward)
        if "cu_seq_lens" in source:
            print("  [PASS] Qwen3_5MoeDecoderLayer.forward 处理 cu_seq_lens")
        else:
            print("  [FAIL] Qwen3_5MoeDecoderLayer.forward 未处理 cu_seq_lens")
            return False

        return True

    except Exception as e:
        print(f"  [FAIL] 导入或检查失败: {e}")
        return False


def test_chunk_gated_delta_rule_varlen():
    """测试 chunk_gated_delta_rule 的变长序列功能"""
    print_section("步骤 3: 测试 GDN 算子变长序列功能")

    try:
        # 尝试导入 verl 的实现
        try:
            from verl.models.transformers import chunk_gated_delta_rule
            print("  [INFO] 使用 verl 的 chunk_gated_delta_rule 实现")
        except ImportError:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule as chunk_gated_delta_rule,
            )
            print("  [INFO] 使用 torch 的 chunk_gated_delta_rule 实现")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        # 测试配置
        seq_len_1, seq_len_2 = 64, 128
        batch_size = 1
        num_heads = 4
        head_dim = 64
        total_len = seq_len_1 + seq_len_2

        torch.manual_seed(42)

        # 创建输入
        q = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        k = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        v = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device)
        g = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device)
        beta = torch.sigmoid(beta)

        cu_seqlens = torch.tensor([0, seq_len_1, total_len], dtype=torch.int32, device=device)

        # 测试 1: 使用 cu_seqlens 计算
        print(f"  测试配置: batch_size={batch_size}, seq_lengths=[{seq_len_1}, {seq_len_2}]")
        print(f"  cu_seqlens: {cu_seqlens.tolist()}")

        out_varlen, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )
        print(f"  [PASS] 变长序列计算成功，输出形状: {out_varlen.shape}")

        # 测试 2: 分别计算每个序列
        q1, k1, v1, g1, beta1 = [x[:, :seq_len_1] for x in (q, k, v, g, beta)]
        q2, k2, v2, g2, beta2 = [x[:, seq_len_1:] for x in (q, k, v, g, beta)]

        out1, _ = chunk_gated_delta_rule(
            q1, k1, v1, g1, beta1,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )
        out2, _ = chunk_gated_delta_rule(
            q2, k2, v2, g2, beta2,
            cu_seqlens=None,
            use_qk_l2norm_in_kernel=True,
        )
        out_separate = torch.cat([out1, out2], dim=1)

        # 比较结果
        diff = (out_varlen.float() - out_separate.float()).abs().max().item()
        print(f"  变长序列 vs 分离计算 最大差异: {diff:.6e}")

        if diff < 1e-2:
            print(f"  [PASS] 结果一致性验证通过 (diff < 1e-2)")
        else:
            print(f"  [FAIL] 结果差异过大 (diff = {diff:.6e})")
            return False

        # 测试 3: 序列间无信息泄漏
        q_modified = q.clone()
        q_modified[:, seq_len_1:] = torch.randn_like(q[:, seq_len_1:])

        out_modified, _ = chunk_gated_delta_rule(
            q_modified, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        diff_seq1 = (out_varlen[:, :seq_len_1].float() - out_modified[:, :seq_len_1].float()).abs().max().item()
        print(f"  序列间信息泄漏检测: {diff_seq1:.6e}")

        if diff_seq1 < 1e-5:
            print(f"  [PASS] 无序列间信息泄漏 (diff < 1e-5)")
        else:
            print(f"  [FAIL] 检测到序列间信息泄漏 (diff = {diff_seq1:.6e})")
            return False

        return True

    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """测试梯度反向传播"""
    print_section("步骤 4: 测试梯度反向传播")

    try:
        try:
            from verl.models.transformers import chunk_gated_delta_rule
        except ImportError:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                torch_chunk_gated_delta_rule as chunk_gated_delta_rule,
            )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16

        seq_len_1, seq_len_2 = 64, 64
        batch_size = 1
        num_heads = 4
        head_dim = 64
        total_len = seq_len_1 + seq_len_2

        torch.manual_seed(42)

        q = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        k = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        v = torch.randn(batch_size, total_len, num_heads, head_dim, dtype=dtype, device=device, requires_grad=True)
        g = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device, requires_grad=True)
        beta = torch.randn(batch_size, total_len, num_heads, dtype=dtype, device=device, requires_grad=True)
        beta = torch.sigmoid(beta)

        cu_seqlens = torch.tensor([0, seq_len_1, total_len], dtype=torch.int32, device=device)

        out, _ = chunk_gated_delta_rule(
            q, k, v, g, beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=True,
        )

        loss = out.float().sum()
        loss.backward()

        tensors = {"q": q, "k": k, "v": v, "g": g, "beta": beta}
        all_passed = True
        for name, tensor in tensors.items():
            has_grad = tensor.grad is not None
            no_nan = not torch.isnan(tensor.grad).any() if has_grad else False
            no_inf = not torch.isinf(tensor.grad).any() if has_grad else False

            if has_grad and no_nan and no_inf:
                print(f"  [PASS] {name}: grad mean={tensor.grad.mean().item():.6e}")
            else:
                print(f"  [FAIL] {name}: grad={has_grad}, no_nan={no_nan}, no_inf={no_inf}")
                all_passed = False

        return all_passed

    except Exception as e:
        print(f"  [FAIL] 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("  GDN 变长序列支持 - 快速验证")
    print("=" * 60)

    results = {}

    # 步骤 1: 检查代码修改
    results["代码修改检查"] = check_code_modifications()

    # 步骤 2: 参数传递测试
    results["参数传递测试"] = test_cu_seqlens_passthrough()

    # 步骤 3: 变长序列功能测试
    results["变长序列功能"] = test_chunk_gated_delta_rule_varlen()

    # 步骤 4: 梯度测试
    results["梯度反向传播"] = test_gradient_flow()

    # 汇总结果
    print_section("验证结果汇总")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {name}")

    all_passed = all(results.values())
    print(f"\n{'='*60}")
    if all_passed:
        print("  所有验证通过！GDN 变长序列支持已正确实施。")
    else:
        print("  部分验证失败，请检查上述 FAIL 项。")
    print(f"{'='*60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
