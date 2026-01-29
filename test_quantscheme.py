import re

attn_sizes = {"llama7b": 4096 * 4096 * 4, 
              "dpsk-moe-16b": 2048 * 2048 * 2 + 2048 * 2048 / 16 * 2, 
              "dpsk-v2-lite": 2048 * 2048 + 2048 * 512 + 
                              512 * 1024 * 2 + 16 * 128 * 64 + 16 * 64 * 64 +
                              2048 * 2048,  ### MLA,
              "llama3-8b": 4096 * 4096 * 2 + 4096 * 1024 * 2, # GQA
              "qwen3-8b": 4096 * 4096 * 2 + 4096 * 1024 * 2, # GQA
              "llama2-13b": 5120 * 5120 * 4, # MHA
              } 
# print(attn_sizes)
shared_sizes = {"llama7b": 0, "llama3-8b": 0, "qwen3-8b": 0, "llama2-13b": 0,
                "dpsk-moe-16b": 10944 * 2048 * 3, 
                "dpsk-v2-lite": 10944 * 2048 * 3, 
                }
ffn_sizes = {"llama7b": 4096 * 11008 * 3, 
             "dpsk-moe-16b": 2048 * 1408 * 3 * 64,
             "dpsk-v2-lite": 2048 * 1408 * 3 * 64,
             "llama3-8b": 4096 * 14336 * 3,
             "qwen3-8b": 4096 * 12288 * 3,
             "llama2-13b": 5120 * 13824 * 3,
             }
layer_nums = {"llama7b": 32, 
              "dpsk-moe-16b": 26,
              "dpsk-v2-lite": 26,
              "llama3-8b": 32,
              "qwen3-8b": 36,
              "llama2-13b": 40,
              }
vol_sizes = {"llama7b": 32000 * 4096, 
             "dpsk-moe-16b": 102400 * 2048,
             "dpsk-v2-lite": 102400 * 2048,
             "llama3-8b": 128256 * 4096,
             "qwen3-8b": 151936 * 4096,
             "llama2-13b": 32000 * 5120,
             }


modeltype = "llama7b"
modeltype = "llama3-8b"
modeltype = "qwen3-8b"
modeltype = "llama2-13b"
qscheme_strs = [
"a8s0m8",
"a8s0m4",
"a4s0m4",
"a4s0m3",
"a4s0m2",
"a3s0m3",
"a3s0m2",
"a2s0m2",
"a4s0m4422",
"a4s0m4332",
"a4s0m4333",
"a4s0m2334",
"a4s0m4222",
"a4s0m4221",
"a4s0m3333",
"a4s0m3322",
"a4s0m3222",
"a4s0m3221",
"a3s0m4422",
"a3s0m4332",
"a3s0m4222",
"a3s0m4221",
"a3s0m3333",
"a3s0m3332",
"a3s0m3322",
"a3s0m3321",
"a3s0m3221",
"a4s0m43333333",
"a3s0m43333333",
"a3s0m43333332",
"a4s0m44433222",
"a3s0m43333222",
]

# modeltype = "dpsk-moe-16b"
# modeltype = "dpsk-v2-lite"
# qscheme_strs = ["a8s8m8",
#                 "a8s4m4",
#                 "a8s4m3",
#                 "a8s3m3",
#                 "a8s4m2",
#                 "a8s2m2",
#                 "a4s4m4",
#                 "a4s4m3",
#                 "a4s4m2",
#                 "a4s3m3",
#                 "a4s3m2",
#                 "a4s2m2",
#     "a8s4m22",
#     "a8s4m42",
#     "a8s4m32",
#     "a8s2m22",
#     "a8s4m3221",
#     "a8s2m3222",
#     "a8s2m2222",
#     "a8s2m3221"]

print(modeltype)
for qscheme_str in qscheme_strs:
    if qscheme_str is not None:
        match = re.search(r'a(\d)s(\d)m(\d+)', qscheme_str)
        qscheme_attn = [int(match.group(1))]
        qscheme_share = [int(match.group(2))]
        ee = match.group(3)
        qscheme_expert = [int(e) for e in ee]

        attn_size = attn_sizes[modeltype]
        ffn_size = ffn_sizes[modeltype]

        final_size = 0
        final_size += vol_sizes[modeltype]
        final_size += attn_size * (qscheme_attn[0] + 0.25)
        final_size += shared_sizes[modeltype] * (qscheme_share[0] + 0.25)

        expert_slice_num = len(qscheme_expert)
        # print(qscheme_attn[0], qscheme_share[0], qscheme_expert)
        for e in qscheme_expert:
            final_size += ffn_size / expert_slice_num * (e + 0.25)

        final_size *= layer_nums[modeltype]
        if modeltype == "dpsk-moe-16b" or modeltype == "dpsk-v2-lite":
            final_size += attn_size * (qscheme_attn[0] + 0.25)
            final_size += shared_sizes[modeltype] * (qscheme_share[0] + 0.25)
        final_size_gb = final_size / 8 / 1024 / 1024 / 1024
        # print(f"{qscheme_str}\t {final_size}b\t {final_size_gb:.4f} GB")
        # print(f"64x2048, S{expert_slice_num}/{expert_slice_num}, a{qscheme_attn[0]}, m{ee}")
        print(f"{final_size_gb :.4f}")
