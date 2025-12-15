FROM python:3.10-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch \
    tensorflow==2.11.0 \
    torch-geometric \
    waymo-open-dataset-tf-2-11-0 \
    tqdm \
    numpy \
    protobuf==3.20.3

COPY . .

CMD ["/bin/bash"]