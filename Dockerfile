# Use the Python version matching your local environment
FROM python:3.13-slim

WORKDIR /app

# Copy the backend requirements and install them
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire backend folder into the container
COPY backend/ ./backend/

# Expose Render's default port
EXPOSE 10000

# Start the FastAPI server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "10000"]
