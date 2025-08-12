# Social Darwin Gödel Machine Docker Image
# Extends the base DGM container with social media capabilities

FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    wget \
    postgresql-client \
    ffmpeg \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /social_dgm

# Copy requirements and install Python dependencies
COPY requirements.txt requirements_social.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements_social.txt

# Copy source code
COPY . .

# Set up git configuration for the container
RUN git config --global user.name "Social DGM Agent"
RUN git config --global user.email "social-dgm@example.com"
RUN git config --global init.defaultBranch main

# Initialize git repository if not already initialized
RUN if [ ! -d .git ]; then git init; fi

# Create necessary directories
RUN mkdir -p /social_dgm/logs \
             /social_dgm/outputs \
             /social_dgm/temp \
             /social_dgm/uploads \
             /social_dgm/cache

# Set environment variables
ENV PYTHONPATH=/social_dgm
ENV PYTHONUNBUFFERED=1
ENV SOCIAL_DGM_HOME=/social_dgm

# Expose ports for web interface (if needed)
EXPOSE 8000 8001 8002

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Default command
CMD ["python", "social_dgm_outer.py", "--help"]