FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install system deps
RUN apt-get update && \
    apt-get install -y python3.10 python3-pip build-essential git graphviz && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install base requirements
RUN pip3 install --no-cache-dir -r requirements.txt

# Install specific versions for compatibility
# Using PyTorch 2.0.1 instead of 2.6+ to avoid unpickling issues with older checkpoints
RUN pip3 install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    torchaudio==2.0.2 \
    --extra-index-url https://download.pytorch.org/whl/cu118 \
    pyarrow==14.0.1 \
    transformers==4.36.2 \
    streamlit==1.29.0 \
    pandas==2.0.3 \
    datasets==2.14.6

# Copy application code
COPY . .

# Set up environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "app.py"]