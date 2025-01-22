# Use official Python runtime
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy dependency file first
COPY requirements.txt requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt


# Copy application source code
COPY . .

# Expose port and run FastAPI server
EXPOSE 8000

# Make the script executable
RUN chmod +x start.sh

# Set the entrypoint to the start script
ENTRYPOINT ["./start.sh"]