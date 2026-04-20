hidden_size = 4096
vocab_size = 151936
num_hidden_layers = 36
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
intermediate_size = 12288

num_full_attn = 27
num_linear_attn = 9

linear_head_k_dim = 128
linear_head_v_dim = 64
linear_num_v_heads = 32

q_size = num_attention_heads * head_dim * 2
k_size = num_key_value_heads * head_dim
v_size = num_key_value_heads * head_dim
o_size = num_attention_heads * head_dim

mlp_N = hidden_size * intermediate_size * 3
attn_linear_N = hidden_size * (q_size + k_size + v_size + o_size)
emd_and_lm_head_N = vocab_size * hidden_size * 2
dense_N = (mlp_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

batch_seqlens_1 = [512, 1024, 2048]
batch_seqlens_2 = [4096, 4096, 4096]

print("=== Qwen3.5 Dense ===")
for batch_seqlens in [batch_seqlens_1, batch_seqlens_2]:
    tokens_sum = sum(batch_seqlens)
    dense_N_flops = 6 * dense_N * tokens_sum

    full_attn_seqlen_square_sum = 0
    linear_attn_seqlen_sum = 0
    for seqlen in batch_seqlens:
        full_attn_seqlen_square_sum += seqlen * seqlen * num_full_attn
        linear_attn_seqlen_sum += seqlen * num_linear_attn

    attn_qkv_flops = 6 * full_attn_seqlen_square_sum * head_dim * num_attention_heads
    linear_attn_flops = 6 * linear_attn_seqlen_sum * linear_head_k_dim * linear_head_v_dim * linear_num_v_heads
    attn_qkv_flops += linear_attn_flops

    total_flops = dense_N_flops + attn_qkv_flops
    print(f"batch_seqlens={batch_seqlens}")
    print(f"  total_flops/1e12={total_flops/1e12}")
    print(f"  total_flops={int(total_flops)}")
    print()

print("=== Qwen3.5 MoE ===")
hidden_size = 3584
vocab_size = 200000
num_hidden_layers = 48
num_attention_heads = 32
num_key_value_heads = 8
head_dim = 128
moe_intermediate_size = 2560
moe_topk = 2
moe_num_expert = 128
shared_expert_intermediate_size = 2560

num_full_attn = 36
num_linear_attn = 12

linear_head_k_dim = 128
linear_head_v_dim = 64
linear_num_v_heads = 32

q_size = num_attention_heads * head_dim * 2
k_size = num_key_value_heads * head_dim
v_size = num_key_value_heads * head_dim
o_size = num_attention_heads * head_dim

moe_gate_N = hidden_size * moe_num_expert
moe_expertmlp_N = hidden_size * moe_intermediate_size * moe_topk * 3
shared_expert_N = hidden_size * shared_expert_intermediate_size * 3
attn_linear_N = hidden_size * (q_size + k_size + v_size + o_size)
emd_and_lm_head_N = vocab_size * hidden_size * 2
moe_N = (moe_gate_N + moe_expertmlp_N + shared_expert_N + attn_linear_N) * num_hidden_layers + emd_and_lm_head_N

for batch_seqlens in [batch_seqlens_1, batch_seqlens_2]:
    tokens_sum = sum(batch_seqlens)
    dense_N_flops = 6 * moe_N * tokens_sum

    full_attn_seqlen_square_sum = 0
    linear_attn_seqlen_sum = 0
    for seqlen in batch_seqlens:
        full_attn_seqlen_square_sum += seqlen * seqlen * num_full_attn
        linear_attn_seqlen_sum += seqlen * num_linear_attn

    attn_qkv_flops = 6 * full_attn_seqlen_square_sum * head_dim * num_attention_heads
    linear_attn_flops = 6 * linear_attn_seqlen_sum * linear_head_k_dim * linear_head_v_dim * linear_num_v_heads
    attn_qkv_flops += linear_attn_flops

    total_flops = dense_N_flops + attn_qkv_flops
    print(f"batch_seqlens={batch_seqlens}")
    print(f"  total_flops/1e12={total_flops/1e12}")
    print(f"  total_flops={int(total_flops)}")
    print()
