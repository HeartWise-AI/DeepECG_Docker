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

# Set entrypoint
ENTRYPOINT ["./run_pipeline.bash"]

# Use CMD to set the defualt mode 
CMD ["full_run"]