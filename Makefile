include .env
export

.PHONY: install build run clean

build:
	@echo "Building Docker image..."
	@./docker/build.sh
	@echo "Build complete. Log saved to $(BUILD_LOG)"

run:
	@echo "Running Docker image..."
	@./docker/run.sh

clean:
	@echo "Cleaning up Docker containers..."
	@docker ps -aq | xargs -r sudo docker rm -f
	@echo "Cleaning up Docker images..."
	@sudo docker images -q $(DOCKER_IMAGE) | xargs -r sudo docker rmi -f

install:
	@echo "Install packages"
	@./docker/install.sh

install-gpu: install
	@pip install --upgrade "jax[cuda110]"==0.2.19 -f https://storage.googleapis.com/jax-releases/jax_releases.html