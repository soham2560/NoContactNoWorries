FROM nvidia/cuda:12.4.1-base-ubuntu22.04

# --- ARGUMENTS ---
ARG USERNAME=container_user
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# --- METADATA / LABELS ---
LABEL org.opencontainers.image.description="NoContactNoWorries Development Container"
LABEL org.opencontainers.image.title="ncnw_dev_container"
LABEL org.opencontainers.image.vendor="Soham Patil"
LABEL org.opencontainers.image.source="https://github.com/soham2560/NoContactNoWorries"
LABEL maintainer="sohampatil45939@gmail.com"
LABEL org.opencontainers.image.licenses="MIT"

# --- ENVIRONMENT SETUP ---
SHELL ["/bin/bash", "-c"]
ENV SHELL=/bin/bash
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive

# --- USER CREATION ---
    RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME
# --- SWITCH TO NON-ROOT USER ---
USER $USERNAME

# --- SYSTEM PACKAGES INSTALLATION ---
RUN sudo apt-get update && sudo apt-get install -q -y --no-install-recommends \
    # System Utilities
    p7zip-full \
    bash-completion \
    build-essential \
    git \
    # Python related
    python3.10 \
    python3.10-dev \
    python3-pip \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/*

# --- PYTHON VERSION MANAGEMENT ---
RUN sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN sudo update-alternatives --config python3

# --- PYTHON PACKAGES INSTALLATION (as non-root user) ---
RUN python3.10 -m pip install --upgrade pip \
    && python3.10 -m pip install --no-cache-dir --user \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

RUN python3.10 -m pip install --no-cache-dir --user \
    timm \
    thop \
    accelerate \
    wandb \
    gdown

# --- ENVIRONMENT VARIABLES FOR USER ---
WORKDIR /home/$USERNAME
ENV PATH="/home/$USERNAME/.local/bin:${PATH}"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:${PATH}