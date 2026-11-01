# Isolated MapTRv2 inference environment (torch 1.9.1 / CUDA 11.1 / mmcv-full 1.4.0).
# It cannot share the BEVFusion .venv (torch 2.0.1 / mmcv 2.x), so it lives here.
#
#   docker build -t maptrv2:cu111 projects/LaneMapping/docker
FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ARG MAPTR_COMMIT=e03f097
ENV DEBIAN_FRONTEND=noninteractive \
    TORCH_CUDA_ARCH_LIST="8.6" \
    FORCE_CUDA=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-dev python3-pip git ninja-build \
        libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && ln -sf /usr/bin/python3.8 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade "pip<24" "setuptools==59.5.0" wheel

RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
        -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install mmcv-full==1.4.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
RUN pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm==0.6.13

RUN git clone -b maptrv2 https://github.com/hustvl/MapTR.git /opt/MapTR \
    && cd /opt/MapTR && git checkout ${MAPTR_COMMIT}

WORKDIR /opt/MapTR/mmdetection3d
RUN pip install "numpy==1.23.5" "numba==0.56.4" "llvmlite==0.39.1" \
        "networkx==2.8.8" "trimesh==3.23.5" "scikit-image==0.19.3" \
        "lyft_dataset_sdk==0.0.8" "plyfile" "tensorboard" "pyquaternion" \
    && python setup.py develop --no-deps \
    && grep -rl "from numba.errors" mmdet3d | xargs sed -i 's/from numba.errors/from numba.core.errors/'

WORKDIR /opt/MapTR/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn
# No GPU is visible during `docker build`; the arch list above targets the A10G.
RUN sed -i 's/torch.cuda.is_available() and CUDA_HOME/CUDA_HOME/' setup.py \
    && python setup.py build install

RUN pip install "shapely==1.8.5" "av2==0.2.1" "nuscenes-devkit==1.1.10" \
        "numpy==1.23.5" "matplotlib<3.8" "opencv-python-headless<4.9" \
        "yapf==0.40.1" "pandas<2.1"

# Upstream bevformer/detectors/bevformer.py imports tkinter.
RUN apt-get update && apt-get install -y --no-install-recommends python3-tk \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH=/opt/MapTR
WORKDIR /workspace
