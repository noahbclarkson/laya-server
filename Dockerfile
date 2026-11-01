# laya-server: Laya behind the TypeSafe System One wire format.
#
#   docker build -t laya-server .                                         CPU, runs anywhere
#   docker build -t laya-server:cuda --build-arg TORCH_INDEX=cu130 .      NVIDIA GPU
#   docker build -t laya-server:rocm --build-arg TORCH_INDEX=rocm7.2 .    AMD GPU, Linux host
#
# The compose files wrap these; see the README.

FROM python:3.15-slim

# Which of PyTorch's wheel indexes to install from: https://download.pytorch.org/whl/<index>
ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.14.0

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/data/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    LAYA_SERVER_HOST=0.0.0.0 \
    LAYA_SERVER_PORT=8765 \
    LAYA_SERVER_JEV_ALIAS=1

# PyTorch first, from the hardware-specific index, so Laya's `torch` requirement
# is already met and pip does not replace it.
RUN pip install --index-url "https://download.pytorch.org/whl/${TORCH_INDEX}" "torch==${TORCH_VERSION}"

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# `video` and `render` membership is granted at run time for AMD GPUs; see compose.rocm.yaml.
RUN useradd --create-home --uid 1000 laya \
    && mkdir -p /data \
    && chown laya /data
COPY server.py check_gpu.py ./

USER laya
# Checkpoints are downloaded here on first start; keep it on a volume.
VOLUME /data
EXPOSE 8765

# The first start downloads about 850 MB per checkpoint, hence the long start period.
HEALTHCHECK --interval=10s --timeout=5s --start-period=30m --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/healthz', timeout=4)"]

CMD ["python", "server.py"]
