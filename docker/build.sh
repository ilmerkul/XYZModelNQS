#!/bin/bash

# Скрипт для сборки Docker-образа
set -e  # Прерывать выполнение при ошибках

# Проверка обязательных переменных окружения
if [ -z "$DOCKERFILE" ] || [ -z "$DOCKER_IMAGE" ] || [ -z "$CONTAINER_WORKDIR" ] || [ -z "$BUILD_LOG" ]; then
  echo "Ошибка: Необходимо задать переменные окружения:"
  echo "  DOCKERFILE, DOCKER_IMAGE, CONTAINER_WORKDIR, BUILD_LOG"
  exit 1
fi

# Проверка существования Dockerfile
if [ ! -f "$DOCKERFILE" ]; then
  echo "Ошибка: Dockerfile не найден по пути: $DOCKERFILE"
  exit 1
fi

echo "Сборка образа $DOCKER_IMAGE..."
echo "Используется Dockerfile: $DOCKERFILE"
echo "Логи сборки будут записаны в: $BUILD_LOG"

# Сборка образа
if ! docker image build -f "$DOCKERFILE" \
  --tag "$DOCKER_IMAGE" \
  . > "$BUILD_LOG" 2>&1; then
  echo "Ошибка сборки образа! Проверьте лог: $BUILD_LOG"
  exit 1
fi

echo "Сборка успешно завершена!"
echo "Собранный образ: $DOCKER_IMAGE"