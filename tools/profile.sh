dcgmi profile --pause
ncu --set full --export ncu/minigpt.ncu-rep python scripts/eval.py --level 3 --task 43 --code_path KernelBench/level3/43_MinGPTCausalAttention.py --profile
