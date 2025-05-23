FROM nvidia/cuda:11.7.1-base-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and pip
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/bin/python

WORKDIR /app

COPY app/requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117 && \
    pip install -r requirements.txt

COPY . .

RUN mkdir -p /app/data/test_infer

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 