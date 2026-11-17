# Dockerfile for Frontend (React + Vite + Tailwind)
# Multi-stage build: builder -> production

# ========================================================================
# STAGE 1: Builder
# ========================================================================
FROM node:22-alpine AS builder

WORKDIR /app

# Copy package files
COPY web/frontend/package*.json ./

# Install dependencies
RUN npm ci --no-audit --no-fund

# Copy source code
COPY web/frontend/ ./

# Build arguments
ARG VITE_API_URL=http://localhost:8000
ENV VITE_API_URL=$VITE_API_URL

# Build production bundle
RUN npm run build

# ========================================================================
# STAGE 2: Production (Nginx)
# ========================================================================
FROM nginx:alpine AS production

# Copy built assets
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy nginx configuration
COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Expose port
EXPOSE 80

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost/ || exit 1

# Run nginx
CMD ["nginx", "-g", "daemon off;"]