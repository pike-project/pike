# destination to rsync data to on local machine
DST1=../KernelBench-data/perlmutter/
DST2=../KernelBench-data/lrc/

mkdir -p $DST1
mkdir -p $DST2

rsync -avz perlmutter:/pscratch/sd/k/kir/llm/KernelBench/data/ $DST1
rsync -avz lrc-xfer:/global/scratch/users/knagaitsev/KernelBench/data/ $DST2
