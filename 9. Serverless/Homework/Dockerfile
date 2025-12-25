# Use the prebuilt lambda image as base
FROM agrigorev/model-2025-hairstyle:v1

# Install required Python packages
RUN pip install --no-cache-dir pillow numpy onnxruntime

# Copy your lambda/homework code
COPY q6.py /var/task/question6.py

# Set working directory
WORKDIR /var/task

# Optional: set default command to run question6.py
CMD ["python3", "q6.py"]