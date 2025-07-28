# Multi-stage Dockerfile for Mini-NanoGPT with CUDA auto-detection
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Create two virtual environments
RUN python3 -m venv /opt/venv-cuda && \
    python3 -m venv /opt/venv-cpu

# 位于中国内地的用户可以取消注释以使用镜像源安装
# Users in mainland China can uncomment the following lines to use mirror sources
# RUN /opt/venv-cuda/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
#     /opt/venv-cpu/bin/pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies in CUDA environment
RUN /opt/venv-cuda/bin/pip install --upgrade pip && \
    /opt/venv-cuda/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
    /opt/venv-cuda/bin/pip install -r requirements.txt

# 位于中国内地的用户可以取消注释以使用清华镜像源安装
# Users in mainland China can uncomment the following lines to use Tsinghua mirror for PyTorch
# RUN /opt/venv-cuda/bin/pip install --upgrade pip && \
#     /opt/venv-cuda/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118 && \
#     /opt/venv-cuda/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install dependencies in CPU environment
RUN /opt/venv-cpu/bin/pip install --upgrade pip && \
    /opt/venv-cpu/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu && \
    /opt/venv-cpu/bin/pip install -r requirements.txt

# 位于中国内地的用户可以取消注释以使用清华镜像源安装PyTorch (CPU版本)
# Users in mainland China can uncomment the following lines to use Tsinghua mirror for PyTorch (CPU version)
# RUN /opt/venv-cpu/bin/pip install --upgrade pip && \
#     /opt/venv-cpu/bin/pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple && \
#     /opt/venv-cpu/bin/pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# Copy application code
COPY . .

# Create startup script
COPY docker/start.sh /start.sh
RUN chmod +x /start.sh

# Create CUDA detection script
COPY docker/detect_cuda.py /detect_cuda.py

# Expose port
EXPOSE 7860

# Set default command
CMD ["/start.sh"]