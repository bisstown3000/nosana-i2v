# torch's cu126 wheels carry their own CUDA runtime, so a plain python base is
# enough - the node supplies the driver and nvidia-container-toolkit, nothing
# else. diffusers 0.42 needs Python >= 3.11, which rules out the ubuntu 22.04
# CUDA images (python 3.10).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/work/cache/hf \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Pinned so a rented node never spends paid GPU seconds resolving a surprise
# release. Bump deliberately, with a benchmark run to back it up.
#
# cu126 rather than the newest CUDA build on purpose: market nodes run whatever
# driver their operator installed, and an older CUDA minor is the safer bet.
RUN pip install --no-cache-dir \
        torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126

# diffusers >= 0.42 is the floor, not a preference: Wan2.2 needs the MoE
# two-transformer path (`boundary_ratio`) and the 5B TI2V `expand_timesteps`
# path, neither of which exists in older releases.
RUN pip install --no-cache-dir \
        diffusers==0.42.1 \
        transformers==5.23.0 \
        accelerate==1.16.0 \
        huggingface_hub==2.4.0 \
        safetensors==0.8.0 \
        sentencepiece==0.2.3 \
        ftfy==6.3.1 \
        imageio==2.38.2 \
        imageio-ffmpeg==0.8.1 \
        pillow==13.0.0 \
        numpy==2.5.5

WORKDIR /work
COPY worker/ /work/worker/
COPY bench/ /work/bench/

# /work/in  (read)  job.json + source stills, local runs only
# /work/out (write) clips + per-clip timing json
# /work/cache       model snapshot; mount it to keep the download one-time
VOLUME ["/work/in", "/work/out", "/work/cache"]

ENTRYPOINT ["python", "/work/worker/i2v.py"]
