# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering
# unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2ForCausalLM, DeepseekV3ForCausalLM

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3_5MoeForCausalLM)
except ImportError:
    pass


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        raise ValueError("The provided model does not have a valid 'model' or 'language_model' attribute.")

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(inner_model, tuple(SUPPORTED_MOE_MODELS)):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ in ("Qwen3MoeLLMForCausalLM", "Qwen3_5MoeForCausalLM"):
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader


def _convert_fused_expert_weights(weights, num_experts):
    """Convert fused expert weights (transformers 5.x) to checkpoint format.

    transformers 5.x stores expert weights as 3D fused tensors:
        mlp.experts.gate_up_proj  (num_experts, 2*intermediate, hidden)
        mlp.experts.down_proj     (num_experts, hidden, intermediate)

    Checkpoint format (what vLLM load_weights expects):
        mlp.experts.0.gate_proj.weight  (intermediate, hidden)
        mlp.experts.0.up_proj.weight    (intermediate, hidden)
        mlp.experts.0.down_proj.weight  (hidden, intermediate)
    """
    fused_count = 0
    passthrough_count = 0
    for name, tensor in weights:
        if "mlp.experts.gate_up_proj" in name and tensor.dim() == 3:
            fused_count += 1
            print(
                f"[Qwen3Moe-FusedExpert] DETECTED fused gate_up_proj: "
                f"{name} shape={tuple(tensor.shape)} -> "
                f"splitting into {num_experts} experts (gate_proj + up_proj)"
            )
            # Split fused gate_up_proj into per-expert gate_proj and up_proj
            gate, up = tensor.chunk(2, dim=1)
            for expert_id in range(num_experts):
                yield (
                    name.replace(
                        "experts.gate_up_proj",
                        f"experts.{expert_id}.gate_proj.weight",
                    ),
                    gate[expert_id],
                )
                yield (
                    name.replace(
                        "experts.gate_up_proj",
                        f"experts.{expert_id}.up_proj.weight",
                    ),
                    up[expert_id],
                )
        elif "mlp.experts.down_proj" in name and tensor.dim() == 3:
            fused_count += 1
            print(
                f"[Qwen3Moe-FusedExpert] DETECTED fused down_proj: "
                f"{name} shape={tuple(tensor.shape)} -> "
                f"splitting into {num_experts} experts (down_proj)"
            )
            # Split fused down_proj into per-expert down_proj
            for expert_id in range(num_experts):
                yield (
                    name.replace(
                        "experts.down_proj",
                        f"experts.{expert_id}.down_proj.weight",
                    ),
                    tensor[expert_id],
                )
        else:
            passthrough_count += 1
            yield (name, tensor)

    if fused_count > 0:
        print(
            f"[Qwen3Moe-FusedExpert] Conversion summary: {fused_count} fused "
            f"tensors split, {passthrough_count} tensors passed through unchanged"
        )
    else:
        print(
            f"[Qwen3Moe-FusedExpert] No fused expert tensors detected "
            f"(all {passthrough_count} tensors passed through unchanged). "
            f"This is expected for checkpoint/safetensors format."
        )


def patch_qwen3_moe_fused_expert_weights(model):
    """Patch Qwen3MoeModel.load_weights to support fused expert format from
    transformers 5.x.

    transformers 5.x refactored Qwen3MoeExperts to use 3D fused tensors
    (experts.gate_up_proj, experts.down_proj) instead of per-expert 2D
    tensors (experts.0.gate_proj.weight, etc). vLLM's Qwen3MoeModel.load_weights
    only understands the checkpoint format, so expert weights are silently
    skipped and the model produces garbage output.

    This patch wraps load_weights with a pre-processing step that splits
    the 3D fused tensors back into per-expert 2D tensors in checkpoint format.
    """
    try:
        from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
    except ImportError:
        return

    # Unwrap ACLGraphWrapper if present
    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable

    inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
    if inner_model is None:
        return

    if not isinstance(inner_model, Qwen3MoeModel):
        return

    if getattr(inner_model, "_fused_expert_patched", False):
        return

    original_load_weights = inner_model.load_weights
    num_experts = inner_model.config.num_experts

    print(
        f"[Qwen3Moe-FusedExpert] Patching Qwen3MoeModel.load_weights "
        f"(num_experts={num_experts}) to handle transformers 5.x fused expert format"
    )

    def patched_load_weights(weights):
        converted = _convert_fused_expert_weights(weights, num_experts)
        return original_load_weights(converted)

    inner_model.load_weights = patched_load_weights
    inner_model._fused_expert_patched = True
