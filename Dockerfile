FROM nvcr.io/nvidia/pytorch:24.02-py3

RUN apt-get update && apt-get install -y \
    git curl build-essential pkg-config libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Clone your repo
RUN git clone https://github.com/Volko61/nanochat-mini.git
WORKDIR /workspace/nanochat-mini

# Copy the 1-GPU script
COPY speedrun_1gpu.sh .
RUN chmod +x speedrun_1gpu.sh

# UV installs here
ENV PATH="/root/.local/bin:${PATH}"

CMD ["bash", "speedrun_1gpu.sh"]
