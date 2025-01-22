# Use official Python runtime
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy files and install dependencies
COPY . .
RUN pip install streamlit requests

# Copy source code
COPY . .

# Expose port and run FastAPI server
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

