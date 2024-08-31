FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy the files into the container
COPY . .

# Create a virtual environment and install required libraries
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Run the bash script
CMD ["./run_pipeline.bash"]
