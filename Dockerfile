# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your analyzer.py script
COPY analyzer.py .

# Create input and output directories for volume mounting
RUN mkdir -p /app/input /app/output

# Set default command (will be overridden by docker run arguments)
ENTRYPOINT ["python", "analyzer.py"]
