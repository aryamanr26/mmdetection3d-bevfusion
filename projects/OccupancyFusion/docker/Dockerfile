# Isolated SparseOcc inference environment.
# Official pins: torch 2.0.0+cu118 / mmcv-full 1.6.0 / mmdet 2.28.2 /
# mmdet3d 1.0.0rc6 / Python 3.8. It cannot share the BEVFusion .venv
# (torch 2.0.1 / mmcv 2.x / mmdet3d 1.1.0), so it lives here.
#
#   docker build -t sparseocc:cu118 projects/OccupancyFusion/docker
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ARG SPARSEOCC_COMMIT=af4d9df83bfd1b82c2b095fca793ff7ba49d5e9f
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    PIP_NO_CACHE_DIR=1 \
    CUDA_HOME=/usr/local/cuda

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git ninja-build wget \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libturbojpeg \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel

RUN pip install torch==2.0.0+cu118 torchvision==0.15.0+cu118 \
        --index-url https://download.pytorch.org/whl/cu118

RUN pip install openmim \
    && mim install mmcv-full==1.6.0 \
    && mim install mmdet==2.28.2 \
    && mim install mmsegmentation==0.30.0 \
    && mim install mmdet3d==1.0.0rc6 \
    && pip install "numpy==1.23.5" "setuptools==59.5.0"

RUN git clone https://github.com/MCG-NJU/SparseOcc.git /opt/SparseOcc \
    && cd /opt/SparseOcc && git checkout ${SPARSEOCC_COMMIT}

WORKDIR /opt/SparseOcc/models/csrc
# No GPU is visible during `docker build`; the arch list above targets the A10G.
RUN python setup.py build_ext --inplace

RUN pip install "nuscenes-devkit==1.1.10" "pyquaternion" \
        "opencv-python-headless<4.9" "Pillow<10" "tqdm" \
        "numpy==1.23.5"

ENV PYTHONPATH=/opt/SparseOcc
WORKDIR /workspace
