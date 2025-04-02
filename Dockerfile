ARG CUDA="11.4.3"
ARG CUDNN="8"
ARG TAG="devel"
ARG OS="ubuntu20.04"
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-${TAG}-${OS}

ENV TZ=Europe/Moscow

RUN apt-get update && \
    apt-get install -y \
        git \
        vim \
        htop \
        python3 \
        python3-pip && \
    rm -rf /var/lib/apt/lists/*

ARG JAX_CUDA_CUDNN="cuda"

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install "jax[$JAX_CUDA_CUDNN]" \
        -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --upgrade poetry

ARG WORKDIR_PATH="/app"
WORKDIR ${WORKDIR_PATH}

COPY Makefile ./
RUN make install

CMD ["/bin/bash"]