DATA_DIR=$(realpath ../KernelBench-data)

rsync -avz --delete perlmutter:/pscratch/sd/k/kir/llm/KernelBench/data/ ../KernelBench-data/perlmutter/
rsync -avz --delete lrc-xfer:/global/scratch/users/knagaitsev/KernelBench/data/ ../KernelBench-data/lrc/
