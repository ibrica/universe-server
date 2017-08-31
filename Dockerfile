FROM ubuntu:16.04

RUN apt-get update \
    && apt-get install -y libav-tools \
    python3-numpy \
    python3-scipy \
    python3-setuptools \
    python3-pip \
    libpq-dev \
    libjpeg-dev \
    curl \
    cmake \
    python3-opengl \
    libboost-all-dev \
    libsdl2-dev \
    wget \
    unzip \
    git \

    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && pip install -U pip

# Install pip 
RUN pip install gym[all]
RUN pip install flask

# Install conda




WORKDIR /usr/local/universe-server/

# Cachebusting
COPY ./setup.py ./

RUN pip install -e .

# Upload our actual code
COPY . ./

# Just in case any python cache files were carried over from the source directory, remove them
RUN py3clean .
