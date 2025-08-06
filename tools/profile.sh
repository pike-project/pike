dcgmi profile --pause
ncu --set full --export ncu/minigpt.ncu-rep python scripts/eval.py --level 3 --task 43 --code_path KernelBench/level3/43_MinGPTCausalAttention.py --profile
# ncu --set full --export ncu/efficientnet-b2.ncu-rep python scripts/eval.py --level 3 --task 24 --code_path KernelBench/level3/24_EfficientNetB2.py --profile
# ncu --set full --export ncu/googlenet-module.ncu-rep python scripts/eval.py --level 3 --task 6 --code_path KernelBench/level3/6_GoogleNetInceptionModule.py --profile

# ncu --set full --export ncu/googlenet-module.sol.ncu-rep python scripts/eval.py --level 3 --task 6 --code_path /pscratch/sd/k/kir/llm/KernelBench-data/h100/level3-metr-1/task6.py --profile
# ncu --set full --export ncu/googlenet-module.sol-no-streams.ncu-rep python scripts/eval.py --level 3 --task 6 --code_path /pscratch/sd/k/kir/llm/KernelBench-data/h100/level3-metr-1/task6_no_streams.py --profile
