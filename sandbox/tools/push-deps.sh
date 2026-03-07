#!/bin/bash
set -euo pipefail

TAG="${1:?Usage: $0 <tag>}"

docker tag kernel-bench-deps:latest "ghcr.io/knagaitsev/kernel-bench-deps:${TAG}"
docker push "ghcr.io/knagaitsev/kernel-bench-deps:${TAG}"
