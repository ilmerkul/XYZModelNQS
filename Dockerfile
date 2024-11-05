FROM nvidia/cuda:11.2.2-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow
RUN apt-get update && apt-get install -y python3.8 python3-pip libmpich-dev
RUN python3.8 -m pip install --upgrade pip wheel setuptools
RUN python3.8 -m pip install --pre --upgrade netket[mpi]

ENV JAX_PLATFORM_NAME="cpu"
ENV TF_CPP_MIN_LOG_LEVEL=3
WORKDIR /project
CMD ["/bin/bash"]
