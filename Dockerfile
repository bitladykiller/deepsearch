# ============================================================
# Deep Research Backend Dockerfile
# 基于 python:3.11-slim，多阶段构建
# ============================================================

FROM python:3.11-slim AS base

# 设置环境变量：禁止 Python 写 .pyc，强制 stdout/stderr 不缓冲
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc default-libmysqlclient-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# ---- 依赖安装层（利用 Docker 缓存） ----
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e . 2>/dev/null || true

# ---- 应用代码层 ----
COPY src/ src/
COPY config.json ./
COPY pyproject.toml ./

# 重新安装以注册 src/ 下的包
RUN pip install --no-cache-dir -e .

# 创建非 root 用户
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    mkdir -p /app/data && \
    chown -R appuser:appgroup /app/data

USER appuser

EXPOSE 8000

CMD ["uvicorn", "deep_research.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
