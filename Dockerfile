# Use an official Python runtime as a parent image
FROM python:3.12.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements first for better caching
COPY Dashboard/requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Dashboard directory contents into the container
COPY Dashboard/ /app/Dashboard/

# Copy the Image directory for logo assets
COPY Image/ /app/Image/

# Set working directory to Dashboard for streamlit
WORKDIR /app/Dashboard

# Expose port 8012 for the Streamlit app
EXPOSE 8012

# Define the command to run the app
CMD ["streamlit", "run", "BTM_analysis.py", "--server.port", "8012", "--server.address", "0.0.0.0"]

