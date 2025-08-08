#!/bin/bash

DEPS_DIR=local/deps

mkdir -p "$DEPS_DIR"
pushd "$DEPS_DIR" || exit 1

if [ ! -d "KernelBenchFiltered" ]; then
    git clone git@github.com:METR/KernelBenchFiltered.git
fi

if [ ! -d "good-kernels" ]; then
    git clone git@github.com:ScalingIntelligence/good-kernels.git
fi
