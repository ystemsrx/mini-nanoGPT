<div align="center">

# 🐳 Mini-NanoGPT Docker 部署指南

[![Docker](https://img.shields.io/badge/Docker-支持-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![CUDA](https://img.shields.io/badge/CUDA-自动检测-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)

<p>
  <a href="README.md">English</a>
  ·
  <a href="README.zh.md"><strong>简体中文</strong></a>
</p>

</div>

---

> [!NOTE]
> 此目录下放的文件用于在容器中运行、初始化。

本项目提供了完整的 Docker 解决方案，支持自动检测 CUDA 并选择相应的 PyTorch 环境。

---

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
docker run --gpus all -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/out:/app/out \
  -v $(pwd)/assets:/app/assets \
  mini-nanogpt

# 运行容器（仅CPU模式）
docker run -p 7860:7860 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/out:/app/out \
  -v $(pwd)/assets:/app/assets \
  mini-nanogpt
```

---

## 🐛 故障排除

<details>
<summary><strong>1. GPU 不被识别</strong></summary>

```bash
# 检查 NVIDIA 驱动
nvidia-smi

# 检查 Docker GPU 支持
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu22.04 nvidia-smi
```

</details>

<details>
<summary><strong>2. 端口被占用</strong></summary>

```bash
# 修改 docker-compose.yml 中的端口映射
ports:
  - "8080:7860"  # 使用 8080 端口
```

</details>

<details>
<summary><strong>3. 内存不足</strong></summary>

```bash
# 检查系统资源
docker stats

# 限制容器内存使用
docker run -m 4g mini-nanogpt
```

</details>

### 查看日志

```bash
# Docker Compose 日志
docker-compose logs -f

# Docker 容器日志
docker logs mini-nanogpt
```

---

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

---

## 📝 环境变量

可以通过环境变量自定义配置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | 服务器绑定地址 |
| `GRADIO_SERVER_PORT` | `7860` | 服务器端口 |
| `PYTHONUNBUFFERED` | `1` | Python 输出缓冲 |
| `MINI_NANOGPT_ENV_TYPE` | `AUTO` | 环境类型：`AUTO`、`CUDA`、`CPU` |

**docker-compose.yml 示例：**

```yaml
environment:
  - GRADIO_SERVER_NAME=0.0.0.0
  - GRADIO_SERVER_PORT=7860
  - PYTHONUNBUFFERED=1
  - MINI_NANOGPT_ENV_TYPE=AUTO
```
