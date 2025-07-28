[English](README.md) | [简体中文](README.zh.md)

The files in this directory are used for running and initializing in containers.

# Mini-NanoGPT Docker Deployment Guide

This project provides a complete Docker solution with automatic CUDA detection and corresponding PyTorch environment selection.

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
docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data mini-nanogpt

# Run container (CPU only mode)
docker run -p 7860:7860 -v $(pwd)/data:/app/data mini-nanogpt
```

## 🐛 Troubleshooting

### Common Issues

1. **GPU Not Recognized**
   ```bash
   # Check NVIDIA drivers
   nvidia-smi
   
   # Check Docker GPU support
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **Port Already in Use**
   ```bash
   # Modify port mapping in docker-compose.yml
   ports:
     - "8080:7860"  # Use port 8080
   ```

3. **Insufficient Memory**
   ```bash
   # Check system resources
   docker stats
   
   # Limit container memory usage
   docker run -m 4g mini-nanogpt
   ```

### View Logs
```bash
# Docker Compose logs
docker-compose logs -f

# Docker container logs
docker logs mini-nanogpt
```

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

## 📝 Environment Variables

You can customize configuration through environment variables:

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
  - PYTHONUNBUFFERED=1
  - MINI_NANOGPT_ENV_TYPE=AUTO  # AUTO, CUDA, CPU
```
