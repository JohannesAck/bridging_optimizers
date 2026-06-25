FROM ubuntu:22.04
WORKDIR /workspace

RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-pip \
    bash-completion \
    build-essential \
    patchelf \
    curl \
    git \
    software-properties-common \
    net-tools \
    vim \
    virtualenv \
    wget \
    xpra \
    htop \
    tmux \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# RUN ln -s /usr/bin/python3 /usr/bin/python

# RUN pip install --upgrade pip

# RUN pip install jupyterlab seaborn matplotlib pandas numpy scikit-learn scipy ipywidgets tqdm
# RUN pip install jax

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
