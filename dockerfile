FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Create a virtual environment and install required libraries
RUN python -m venv venv && \
    . venv/bin/activate && \
    pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Run the bash script
CMD ["./run_pipeline.bash"]
