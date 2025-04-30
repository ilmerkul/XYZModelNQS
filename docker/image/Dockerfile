ARG CUDA="12.8.1"
ARG TAG="devel"
ARG OS="ubuntu"
ARG VERSION_OS="24.04"

#FROM nvidia/cuda:${CUDA}-cudnn-${TAG}-${OS}${VERSION_OS}
FROM ${OS}:${VERSION_OS}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

ARG CONTAINER_WORKDIR="/home/ubuntu/project"
ENV CONTAINER_WORKDIR=${CONTAINER_WORKDIR}
ENV VENV_PATH="${CONTAINER_WORKDIR}/.venv"

ENV JAX_CUDA_CUDNN="cuda"

# Install essentials
RUN apt-get update \
	&& apt-get upgrade -y \
	&& apt-get install -y \
		apt-utils

RUN apt-get install -y htop \
                        python3 \
                        python3-pip \
                        python3-venv \
                        python3-dev \
                        python3-full

COPY ./docker/install.sh /tmp/install.sh
RUN chmod +x /tmp/install.sh
RUN ./tmp/install.sh ${VENV_PATH}

WORKDIR ${CONTAINER_WORKDIR}
VOLUME ${CONTAINER_WORKDIR}

CMD [ "/bin/bash" ]