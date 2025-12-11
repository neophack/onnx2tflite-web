FROM python:3.10.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
RUN pip install --no-cache-dir \
    flask==3.0.0 \
    werkzeug==3.0.1 \
    --extra-index-url https://eiq.nxp.com/repository/ eiq-onnx2tflite==0.7.0 \
    onnx==1.17.0 \
    netron==8.7.7 \
    pillow==10.1.0 \
    tensorflow==2.16.1 \
    onnxruntime==1.21.1 \
    numpy==1.24.3

RUN apt-get update && apt-get install -y vim-common  && rm -rf /var/lib/apt/lists/*
# 创建必要的目录
RUN mkdir -p /app/uploads /app/outputs /app/static /app/templates

# 复制应用文件

COPY templates/ /app/templates/
COPY static/ /app/static/
COPY app.py /app/

# 设置环境变量
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# 暴露端口
EXPOSE 5000

# 启动应用
CMD ["python", "app.py"]