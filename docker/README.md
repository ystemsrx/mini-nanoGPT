<div align="center">

# 🐳 Mini-NanoGPT Docker Deployment Guide

[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-Auto--Detect-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<p>
  <a href="README.md"><strong>English</strong></a>
  ·
  <a href="README.zh.md">简体中文</a>
</p>

</div>

---

> [!NOTE]
> The files in this directory are used for running and initializing in containers.

This project provides a complete Docker solution with automatic CUDA detection and corresponding PyTorch environment selection.

---

## 🚀 Quick Start

### Using Docker Compose (Recommended)

```bash
# Start container in foreground
docker-compose up --build

# Stop services
docker-compose down
```

### Using Docker Commands

```bash
# Build image
docker build -t mini-nanogpt .

# Run container (auto GPU detection)
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/out:/app/out \
  -v $(pwd)/assets:/app/assets \
  mini-nanogpt

# Run container (CPU only mode)
docker run -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/out:/app/out \
  -v $(pwd)/assets:/app/assets \
  mini-nanogpt
```

---

## 🐛 Troubleshooting

<details>
<summary><strong>1. GPU Not Recognized</strong></summary>

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

</details>

<details>
<summary><strong>2. Port Already in Use</strong></summary>

```bash
# Modify port mapping in docker-compose.yml
ports:
  - "8080:7860"  # Use port 8080
```

</details>

<details>
<summary><strong>3. Insufficient Memory</strong></summary>

```bash
# Check system resources
docker stats

# Limit container memory usage
docker run -m 4g mini-nanogpt
```

</details>

### View Logs

```bash
# Docker Compose logs
docker-compose logs -f

# Docker container logs
docker logs mini-nanogpt
```

---

## 🔄 Updates and Maintenance

```bash
# Rebuild image
docker-compose build --no-cache

# Clean unused images
docker image prune

# Complete reset
docker-compose down
docker system prune -a
```

---

## 📝 Environment Variables

You can customize configuration through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Server bind address |
| `GRADIO_SERVER_PORT` | `7860` | Server port |
| `PYTHONUNBUFFERED` | `1` | Python output buffering |
| `MINI_NANOGPT_ENV_TYPE` | `AUTO` | Environment type: `AUTO`, `CUDA`, `CPU` |

**Example docker-compose.yml:**

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
  - PYTHONUNBUFFERED=1
  - MINI_NANOGPT_ENV_TYPE=AUTO
```
