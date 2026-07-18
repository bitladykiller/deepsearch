# ============================================================
# Deep Research Frontend Dockerfile
# 构建阶段：Node 20 编译 Vue SPA
# 运行阶段：Nginx 提供静态文件 + 反向代理 API
# ============================================================

# ---- 构建阶段 ----
FROM node:20-alpine AS builder

WORKDIR /app

# 先复制依赖文件，利用缓存
COPY package.json package-lock.json ./
RUN npm ci

# 复制源码并构建
COPY . .
RUN npm run build-only

# ---- 运行阶段 ----
FROM nginx:alpine AS runtime

# 移除默认配置
RUN rm /etc/nginx/conf.d/default.conf

# 复制自定义 Nginx 配置
COPY nginx.conf /etc/nginx/conf.d/default.conf

# 复制构建产物
COPY --from=builder /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
