#!/bin/bash
set -o errexit
export DOCKER_BUILDKIT=1
export PROGRESS_NO_TRUNC=1

docker build --tag xyz_model_nqs \
    --build-arg CUDA="11.4.3" \
    --build-arg OS="ubuntu20.04" \
    --build-arg JAX_CUDA_CUDNN="cuda11_cudnn82"