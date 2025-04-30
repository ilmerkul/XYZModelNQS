#!/bin/bash

# Скрипт для запуска Docker-образа
set -e  # Прерывать выполнение при ошибках

# Проверка обязательных переменных окружения
if [ -z "$PROJECT_DIR" ] || [ -z "$CONTAINER_WORKDIR" ] || [ -z "$DOCKER_IMAGE" ]; then
  echo "Ошибка: Необходимо задать переменные окружения:"
  echo "  PROJECT_DIR, CONTAINER_WORKDIR, DOCKER_IMAGE"
  exit 1
fi

# Запуск контейнера
docker run -it \
  --env CUDA_VISIBLE_DEVICES=-1 \
  --volume="$PROJECT_DIR/project:$CONTAINER_WORKDIR/project" \
  --volume="$PROJECT_DIR/poetry.lock:$CONTAINER_WORKDIR/poetry.lock" \
  --volume="$PROJECT_DIR/pyproject.toml:$CONTAINER_WORKDIR/pyproject.toml" \
  "$DOCKER_IMAGE"