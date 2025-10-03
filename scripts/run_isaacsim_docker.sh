#!/usr/bin/env bash
set -euo pipefail

# 1) подтягиваем контейнер (пример — Isaac Sim 5.0 streaming)
IMAGE="nvcr.io/nvidia/isaac-sim:5.0.0"

# 2) путь к твоему репо на хосте
REPO_HOST="$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

# 3) каталоги для кэшей/логов (персистентные)
mkdir -p ~/docker/isaac-sim/{cache/kit,cache/ov,cache/pip,cache/glcache,cache/computecache,logs,data,documents}

docker pull "$IMAGE"

# 4) запускаем контейнер, монтируем репо в /opt/isaac_hydra_ext и сразу ставим его
docker run --name isaac-sim-appo --rm -it \
  --gpus all --runtime=nvidia --network=host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y \
  -e OMNI_KIT_ACCEPT_EULA=Y \
  -v ~/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
  -v ~/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
  -v ~/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
  -v ~/docker/isaac-sim/cache/glcache:/root/.cache/nvidia/GLCache:rw \
  -v ~/docker/isaac-sim/cache/computecache:/root/.nv/ComputeCache:rw \
  -v ~/docker/isaac-sim/logs:/root/.nvidia-omniverse/logs:rw \
  -v ~/docker/isaac-sim/data:/root/.local/share/ov/data:rw \
  -v ~/docker/isaac-sim/documents:/root/Documents:rw \
  -v "${REPO_HOST}":/opt/isaac_hydra_ext:rw \
  "$IMAGE" bash -lc "
    bash /opt/isaac_hydra_ext/scripts/install_inside_container.sh && \
    python -m isaac_hydra_ext.appo_runner env=isaac_go1_nav experiment.name=docker_gym_test
  "

