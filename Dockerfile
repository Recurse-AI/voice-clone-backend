FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create temp directory
RUN mkdir -p /tmp

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH="/app:$PYTHONPATH"
ENV CUDA_AVAILABLE=true

# Run the application
CMD ["python", "main.py"] 