# Dockerfile
# Use the official Python image from Docker Hub
FROM python:3.12.5

# Set the working directory in the container
WORKDIR /app

# Copy the model and the app code
COPY models/hgbt_final.joblib /models/hgbt_final.joblib
COPY app /app

# Install dependencies
RUN pip install --no-cache-dir -r /app/requirements.txt

# Expose the port that FastAPI runs on
EXPOSE 8000

# Run the FastAPI server with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]