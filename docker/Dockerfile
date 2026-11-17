FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt update && DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
                   apt install -y --allow-unauthenticated \
                                  wget git vim build-essential \
                                  libosmesa6-dev libglew-dev \
                                  glibc-source unzip \
                                  python3-dev python3-pip patchelf \
                                  ffmpeg p7zip-full\
            && apt clean 

RUN python3 -m pip install pip --upgrade pip

RUN pip3 install --no-cache-dir \
                 timm \
                 thop \
                 accelerate \
                 wandb \
                 gdown \
                 tqdm \
                 matplotlib \
                 opencv-python \
                 opencv-python-headless \