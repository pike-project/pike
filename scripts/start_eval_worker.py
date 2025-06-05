import time

# The eval worker waits for new kernel tasks to arrive, then compiles and runs them

# it should make use of recv() and send(), possibly in some library since the sampler
# will also need to send the kernel tasks, then receive the result back from the eval worker

# recv should work by watching a directory for changes in a loop (may not be able to use watchdog due to NFS)
# and when a non-tmp file is found it processes it

# send should work by making a file.json.tmp, and then moving file.json.tmp to file.json only when the tmp
# file is confirmed to be written to disk

# the worker recv will default to "/input" (or you can pass in a path to an input dir)
# the worker send will default to "/output" (or you can pass in a path to an output dir)

# workers will need to have ids, then they will only get their dir exposed to them

# e.g.
#
# KernelBench-data/workers/0/input -> /input
# KernelBench-data/workers/0/output -> /output

# the worker is then responsible for load balancing the evaluations onto the GPUs that it has available to it

# we only technically need to evaluate the reference pytorch implementation once for each task, but the evaluation may
# not be the bottleneck of this anyway (the LLM sampling might be the bottleneck)

def main():
    while True:
        print("Eval worker running...")
        time.sleep(10)

main()
