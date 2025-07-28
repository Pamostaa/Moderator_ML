# Use a PyTorch base image with CUDA support (adjust version as needed; this uses Python 3.11 under the hood, but compatible with your code)
# If you don't need GPU support, you can switch to a CPU-only image like FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime but remove CUDA parts.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# Set working directory
WORKDIR /app

# Copy your script into the container (named api.py)
COPY api.py /app/api.py

# Install additional dependencies, including sentencepiece which is required for CamembertTokenizer
RUN pip install --no-cache-dir transformers pymongo gdown sentencepiece

# Expose no ports since this is a worker script, not an API server

# Run the script
CMD ["python", "api.py"]