# Sandbox Containers

To evaluate solutions, the evaluation worker must execute untrusted, LLM-generated code. As such, sandboxing the evaluation worker is critical. In general, one should instantly be wary of any agent framework which does not sandbox the execution of LLM-generated code.

## Container Config

The current container config tries to reduce the attack surface, in case the LLM generates malicious code which attempts a container breakout.

A few notable design choices are:

- drop all capabilities
- no network (agent framework communicates with worker via fs binding mounts)
- read-only filesystem where possible

## Managing Images

To build the container:

```bash
./sandbox/tools/build-deps.sh
```

To push the container:

```bash
# to push the built image (if you need to)
docker login docker.io
docker tag kernel-bench-deps:latest docker.io/<username>/kernel-bench-deps:<tag>
docker push docker.io/<username>/kernel-bench-deps:<tag>
```

Attach to running docker container:

```bash
docker exec -it <id> /bin/bash
```

### Apptainer Notes

Apptainer:

To fetch an image with a particular tag, do the following

```bash
apptainer registry login --username <username> docker://docker.io
APPTAINER_TMPDIR=/path/to/tmp APPTAINER_CACHEDIR=/path/to/cache apptainer pull kernel-bench-deps.sif docker://docker.io/<username>/kernel-bench-deps:<tag>
```

## Sandbox Alternatives

A number of sandboxing alternatives outside of containers exist.

Better than using a container is using a virtual machine, but this can incur some runtime overhead. Useful tools for running VMs:

- QEMU
- Firecracker: https://github.com/firecracker-microvm/firecracker

A few other notable approaches (usually not as good as a VM, have some tradeoffs relative to plain Docker containers):

- gVisor: https://github.com/google/gvisor
- firejail: https://github.com/netblue30/firejail
- nsjail: https://github.com/google/nsjail
- bubblewrap: https://github.com/containers/bubblewrap
