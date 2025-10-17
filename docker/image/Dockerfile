ARG CUDA="13.0.1"
ARG TAG="devel"
ARG OS="ubuntu"
ARG VERSION_OS="24.04"

FROM nvidia/cuda:${CUDA}-cudnn-${TAG}-${OS}${VERSION_OS}
#FROM ${OS}:${VERSION_OS}

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Moscow

ENV CONTAINER_WORKDIR /home/ubuntu/project
ENV VENV_PATH="${CONTAINER_WORKDIR}/venv"

ENV JAX_CUDA_CUDNN="cuda"

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

WORKDIR ${CONTAINER_WORKDIR}
COPY ./docker/install.sh /tmp/install.sh
#COPY ./poetry.lock .
COPY ./pyproject.toml .
RUN chmod +x /tmp/install.sh
RUN /tmp/install.sh ${VENV_PATH}

COPY ./docker/cmd.sh /tmp/cmd.sh

VOLUME ${CONTAINER_WORKDIR}

ENTRYPOINT ["/tmp/cmd.sh"]
CMD [ "/bin/bash" ]