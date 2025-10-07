# Use official Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Optional: Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose port for Streamlit
EXPOSE 8501

# Default command to run the app
CMD ["sh", "-c", "python run.py & streamlit run streamlit_app.py"]
