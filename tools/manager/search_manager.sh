srun --account=ac_binocular \
    --partition=lr8 \
    --mincpus=64 \
    --mem=64G \
    --nodes=1 \
    --qos=lr8_normal \
    --time=72:0:0 \
    --pty \
    python -u tools/manager/search_manager.py
