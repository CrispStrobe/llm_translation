# Translate a specific dataset with facebook's wmt21 model to german
# everything is hardcoded and done in the main.py script.
# we just need to set one environment variable: HUGGINGFACE_TOKEN
# so you coud run it like this:
# docker run --rm -e HUGGINGFACE_TOKEN=your_token_here crispstrobe/wmt21:latest 

# Uses an official NVIDIA CUDA runtime base image with Ubuntu 20.04
FROM nvidia/cuda:12.4.1-runtime-ubuntu20.04

# Set the working directory in the container to /app
WORKDIR /app

# Set timezone:
ARG CONTAINER_TIMEZONE
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone

# Install tzdata before other dependencies
RUN apt-get update && apt-get install -y tzdata

# Install necessary packages including cmake
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    curl \
    build-essential \
    cmake \
    && curl https://sh.rustup.rs -sSf | sh -s -- -y \
    && . $HOME/.cargo/env \
    && echo 'source $HOME/.cargo/env' >> $HOME/.bashrc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add Cargo's bin directory to the PATH explicitly
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy the Python requirements file into the container at /app
COPY requirements.txt /app/

# Use a new shell to ensure the environment is activated
SHELL ["/bin/bash", "-c"]

# Install required Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . /app

# Command to run on container start
CMD ["python3", "main.py"]
