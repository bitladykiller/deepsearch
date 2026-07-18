# 09 - Docker 部署

## 1. 概述

Deep Research 采用 Docker Compose 实现完全容器化部署，包含 5 个服务、4 个命名卷和 1 个自定义网络。

### 1.1 服务架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Docker Compose                             │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │   MySQL     │  │    Redis     │  │      Milvus          │   │
│  │  8.0.30     │  │  7-alpine    │  │  v2.6.14             │   │
│  │             │  │              │  │                      │   │
│  │  端口:3306  │  │  端口:6379   │  │  端口:19530          │   │
│  │  (仅内部)   │  │  (仅内部)    │  │  (仅内部)            │   │
│  └─────────────┘  └──────────────┘  └──────────────────────┘   │
│        │                │                     │                 │
│        └────────────────┼─────────────────────┘                 │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐   │
│  │                  Backend                                  │   │
│  │              FastAPI + Uvicorn                            │   │
│  │                                                          │   │
│  │   Dockerfile 构建                                        │   │
│  │   python:3.11-slim                                       │   │
│  │   端口: 8080:8000                                        │   │
│  └──────────────────────┬───────────────────────────────────┘   │
│                         │                                       │
│  ┌──────────────────────┴───────────────────────────────────┐   │
│  │                  Frontend                                 │   │
│  │            Nginx + Vue SPA                                │   │
│  │                                                          │   │
│  │   多阶段构建                                             │   │
│  │   node:20-alpine → nginx:alpine                          │   │
│  │   端口: 3000:80                                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  网络: deep_research_net (bridge)                               │
│  卷:   mysql_data, redis_data, milvus_data, app_data            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 对外端口

| 服务 | 对外端口 | 内部端口 | 说明 |
|------|----------|----------|------|
| frontend | **3000** | 80 | 用户访问入口 |
| backend | **8080** | 8000 | API 端口（可选） |
| mysql | ❌ | 3306 | 仅内部访问 |
| redis | ❌ | 6379 | 仅内部访问 |
| milvus | ❌ | 19530 | 仅内部访问 |

### 1.3 命名卷

| 卷名 | 挂载路径 | 用途 |
|------|----------|------|
| `mysql_data` | `/var/lib/mysql` | MySQL 数据持久化 |
| `redis_data` | `/data` | Redis 数据持久化 |
| `milvus_data` | `/var/lib/milvus` | Milvus 向量数据持久化 |
| `app_data` | `/app/data` | 应用运行时数据 |

### 1.4 自定义网络

```yaml
networks:
  deep_research_net:
    driver: bridge
```

所有服务通过 `deep_research_net` 网络通信，使用 Docker 内部 DNS 解析主机名（如 `mysql`、`redis`、`milvus`）。

---

## 2. docker-compose.yml 详解

### 2.1 MySQL 服务

```yaml
mysql:
  image: mysql:8.0.30
  pull_policy: if_not_present
  container_name: deepresearch-mysql
  volumes:
    - mysql_data:/var/lib/mysql
  environment:
    MYSQL_ROOT_PASSWORD: ${MYSQL_ROOT_PASSWORD:-deepresearch123}
    MYSQL_DATABASE: ${MYSQL_DATABASE:-deepresearch}
    MYSQL_USER: ${MYSQL_USER:-deepresearch}
    MYSQL_PASSWORD: ${MYSQL_PASSWORD:-deepresearch123}
  command: --default-authentication-plugin=mysql_native_password --character-set-server=utf8mb4 --collation-server=utf8mb4_unicode_ci
  networks:
    - deep_research_net
  healthcheck:
    test: ["CMD", "mysqladmin", "ping", "-h", "localhost", "-u", "root", "-p${MYSQL_ROOT_PASSWORD:-deepresearch123}"]
    interval: 10s
    timeout: 5s
    retries: 10
    start_period: 30s
  restart: unless-stopped
```

**关键配置**：
- `pull_policy: if_not_present`：优先使用本地已有的镜像，不强制拉取
- `command`：使用 `mysql_native_password` 认证插件（兼容 pymysql），设置 UTF-8 编码
- `healthcheck`：通过 `mysqladmin ping` 检查 MySQL 是否就绪，最多等待 30 秒启动期
- `restart: unless-stopped`：除非手动停止，否则总是重启

### 2.2 Redis 服务

```yaml
redis:
  image: redis:7-alpine
  pull_policy: if_not_present
  container_name: deepresearch-redis
  volumes:
    - redis_data:/data
  networks:
    - deep_research_net
  healthcheck:
    test: ["CMD", "redis-cli", "ping"]
    interval: 10s
    timeout: 5s
    retries: 5
  restart: unless-stopped
```

### 2.3 Milvus 服务

```yaml
milvus:
  image: milvusdb/milvus:v2.6.14
  pull_policy: if_not_present
  container_name: deepresearch-milvus
  volumes:
    - milvus_data:/var/lib/milvus
  environment:
    ETCD_USE_EMBED: "true"
    COMMON_STORAGETYPE: local
  networks:
    - deep_research_net
  healthcheck:
    test: ["CMD-SHELL", "curl -f http://localhost:9091/healthz || exit 1"]
    interval: 15s
    timeout: 10s
    retries: 5
    start_period: 30s
  restart: unless-stopped
```

**关键配置**：
- `ETCD_USE_EMBED: "true"`：使用嵌入式 etcd（单机模式，无需外部 etcd）
- `COMMON_STORAGETYPE: local`：使用本地存储（无需 MinIO）
- `healthcheck`：通过 `/healthz` 端点检查 Milvus 是否就绪

### 2.4 Backend 服务

```yaml
backend:
  build:
    context: .
    dockerfile: Dockerfile
    pull: if_not_present
  container_name: deepresearch-backend
  volumes:
    - ./config.json:/app/config.json:ro
    - ./.env:/app/.env:ro
    - app_data:/app/data
  networks:
    - deep_research_net
  depends_on:
    mysql:
      condition: service_healthy
    redis:
      condition: service_healthy
    milvus:
      condition: service_healthy
  environment:
    MYSQL_DSN: "mysql+pymysql://${MYSQL_USER:-deepresearch}:${MYSQL_PASSWORD:-deepresearch123}@mysql:3306/${MYSQL_DATABASE:-deepresearch}?charset=utf8mb4"
    REDIS_URL: "redis://redis:6379"
    CHECKPOINTER_BACKEND: redis
    SHORT_TERM_BACKEND: mysql
    LONG_TERM_BACKEND: mysql
    MILVUS_HOST: milvus
    MILVUS_PORT: 19530
  ports:
    - "8080:8000"
  restart: unless-stopped
```

**关键配置**：
- `depends_on` + `condition: service_healthy`：等待所有基础设施服务健康检查通过后才启动
- `volumes`：挂载 `config.json` 和 `.env`（只读 `:ro`）
- `environment`：覆盖配置文件中的连接地址为 Docker 内部主机名
- `ports`：将容器内部 8000 端口映射到宿主机 8080

### 2.5 Frontend 服务

```yaml
frontend:
  build:
    context: ./front/agent_front
    dockerfile: Dockerfile
    pull: if_not_present
  container_name: deepresearch-frontend
  networks:
    - deep_research_net
  depends_on:
    - backend
  ports:
    - "3000:80"
  restart: unless-stopped
```

---

## 3. Dockerfile 详解

### 3.1 后端 Dockerfile

```dockerfile
FROM python:3.11-slim

# 系统依赖
RUN apt-get update && \
    apt-get install -y --no-install-recommends default-libmysqlclient-dev build-essential && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python 依赖
COPY pyproject.toml requirements.txt ./
RUN pip install --no-cache-dir -e . || pip install --no-cache-dir -r requirements.txt

# 应用代码
COPY src/ src/
COPY config.json ./
COPY .env* ./

# 非 root 用户
RUN useradd --create-home appuser
USER appuser

CMD ["uvicorn", "deep_research.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**关键步骤**：
1. 基于 `python:3.11-slim`（轻量级 Python 镜像）
2. 安装 `default-libmysqlclient-dev`（MySQL 客户端库，pymysql 需要）
3. 先复制 `pyproject.toml` 和 `requirements.txt`（利用 Docker 层缓存）
4. 安装依赖后再复制源码（源码变更不会触发依赖重新安装）
5. 创建非 root 用户运行应用（安全最佳实践）
6. 使用 uvicorn 启动 FastAPI 应用

### 3.2 前端 Dockerfile（多阶段构建）

```dockerfile
# 构建阶段
FROM node:20-alpine AS builder
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build

# 运行阶段
FROM nginx:alpine
COPY --from=builder /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**多阶段构建优势**：
- 构建阶段包含 Node.js 和 node_modules（~200MB）
- 运行阶段只包含 Nginx 和静态文件（~20MB）
- 最终镜像体积大幅减小

---

## 4. Nginx 配置

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }

    # API 反向代理
    location /api/ {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE 支持
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 300s;
    }

    # 健康检查代理
    location /health {
        proxy_pass http://backend:8000;
    }
}
```

**关键配置**：
- `try_files $uri $uri/ /index.html`：Vue SPA 的 fallback，所有非文件请求都返回 `index.html`
- `proxy_buffering off`：禁用代理缓冲，支持 SSE（Server-Sent Events）流式输出
- `proxy_read_timeout 300s`：将代理读取超时设为 5 分钟（研究任务可能耗时较长）

---

## 5. .dockerignore

```
.git
.env
__pycache__
*.pyc
.pytest_cache
.mypy_cache
data/
node_modules/
front/agent_front/node_modules/
front/agent_front/dist/
```

排除不需要进入镜像的文件，减小构建上下文大小。

---

## 6. 部署命令

### 6.1 首次部署

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env，填写 DASHSCOPE_API_KEY 和 BOCHA_API_KEY

# 2. 构建并启动所有服务
docker compose up -d --build

# 3. 查看服务状态
docker compose ps

# 4. 查看日志
docker compose logs -f backend
```

### 6.2 日常操作

```bash
# 启动所有服务
docker compose up -d

# 停止所有服务
docker compose down

# 重启某个服务
docker compose restart backend

# 查看某个服务的日志
docker compose logs -f mysql

# 进入容器
docker compose exec backend bash

# 更新镜像后重建
docker compose up -d --build
```

### 6.3 数据持久化

命名卷在 `docker compose down` 后仍然保留。要完全清理数据：

```bash
# 停止并删除所有卷
docker compose down -v
```

---

## 7. 环境变量优先级（Docker 环境）

在 Docker 环境中，环境变量的优先级链：

```
Docker Compose environment 字段  >  .env 文件  >  config.json  >  代码默认值
```

Docker Compose 中的 `environment` 字段会覆盖 `.env` 文件中的值。例如：

```yaml
# docker-compose.yml
environment:
  MYSQL_DSN: "mysql+pymysql://deepresearch:deepresearch123@mysql:3306/deepresearch"
```

这会覆盖 `.env` 文件中的 `MYSQL_DSN`，确保后端连接的是 Docker 内部的 MySQL 而非 localhost。
