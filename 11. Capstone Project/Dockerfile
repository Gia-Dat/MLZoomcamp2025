# 1. Use a slim Python image to keep the size small
FROM python:3.10-slim

# 2. Install system dependencies required for image processing (OpenCV/Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory inside the container
WORKDIR /app

# 4. Copy requirements first to leverage Docker caching
COPY requirements.txt .

# 5. Install Python dependencies
# We use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copy the model file and the application code
# Ensure animal_model.pth is in your current local folder
COPY animal_model.pth .
COPY predict.py .

# 7. Expose the port FastAPI will run on
EXPOSE 8080

# 8. Command to run the application
CMD ["uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "8080"]