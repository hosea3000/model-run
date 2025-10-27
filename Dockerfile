# 使用官方 Python 运行时作为基础镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# 设置 HuggingFace 缓存目录
ENV HF_HOME=/app/.cache/huggingface
# 设置模型预下载目录
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# 复制依赖文件
COPY requirements.txt .

# 安装系统依赖和 Python 包
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# 创建缓存目录
RUN mkdir -p /app/.cache/huggingface

# 复制项目文件
COPY . .

# 预下载模型（在容器启动前）
RUN python -c "import sys; sys.path.append('/app'); from image_vector_extractor import ImageVectorExtractor; print('Pre-downloading model...'); extractor = ImageVectorExtractor(); print('Model pre-downloading completed successfully')"

# 暴露端口
EXPOSE 8000

# 设置启动命令
CMD ["python", "main.py"]