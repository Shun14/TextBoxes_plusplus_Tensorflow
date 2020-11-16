FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && \
    apt-get install -y \
    nano less tree \
    python3.5 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY requirements.txt .


RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
# RUN =/usr/local/cuda/lib64/stubs:/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${PATH}"


