[English](README.md) | [简体中文](README.zh.md)

此目录下放的文件用于在容器中运行、初始化。

# Mini-NanoGPT Docker 部署指南

本项目提供了完整的 Docker 解决方案，支持自动检测 CUDA 并选择相应的 PyTorch 环境。

## 🚀 快速开始

### 使用 Docker Compose（推荐）

```bash
# 前台启动容器
docker-compose up --build

# 停止服务
docker-compose down
```

### 使用 Docker 命令

```bash
# 构建镜像
docker build -t mini-nanogpt .

# 运行容器（自动检测GPU）
docker run --gpus all -p 7860:7860 -v $(pwd)/data:/app/data mini-nanogpt

# 运行容器（仅CPU模式）
docker run -p 7860:7860 -v $(pwd)/data:/app/data mini-nanogpt
```

## 🐛 故障排除

### 常见问题

1. **GPU 不被识别**
   ```bash
   # 检查 NVIDIA 驱动
   nvidia-smi
   
   # 检查 Docker GPU 支持
   docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
   ```

2. **端口被占用**
   ```bash
   # 修改 docker-compose.yml 中的端口映射
   ports:
     - "8080:7860"  # 使用 8080 端口
   ```

3. **内存不足**
   ```bash
   # 检查系统资源
   docker stats
   
   # 限制容器内存使用
   docker run -m 4g mini-nanogpt
   ```

### 查看日志
```bash
# Docker Compose 日志
docker-compose logs -f

# Docker 容器日志
docker logs mini-nanogpt
```

## 🔄 更新和维护

```bash
# 重新构建镜像
docker-compose build --no-cache

# 清理未使用的镜像
docker image prune

# 完全重置
docker-compose down
docker system prune -a
```

## 📝 环境变量

可以通过环境变量自定义配置：

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
  - PYTHONUNBUFFERED=1
  - MINI_NANOGPT_ENV_TYPE=AUTO  # AUTO, CUDA, CPU
```
