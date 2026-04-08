FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file if it exists
COPY requirements.txt* ./

# Install dependencies if requirements.txt exists
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install additional ML dependencies
RUN pip install --no-cache-dir fastapi uvicorn numpy torch scikit-learn transformers

# Copy the rest of the code
COPY . .

# Expose port 8000 for the inference service
EXPOSE 8000

# Run inference.py as the entry point
CMD ["python", "inference.py"]