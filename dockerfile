FROM python:3.10

# Set the working directory
WORKDIR /app

# Install git and other necessary packages
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*
    
# Copy the files into the container
COPY requirements.txt .

# Create a virtual environment and install required libraries
RUN pip install -r requirements.txt

# Clone the fairseq-signals repository and install it
RUN git clone https://github.com/HeartWise-AI/fairseq-signals && \
    cd fairseq-signals && \
    pip install --editable ./    

# Copy the rest of the application code into the container
COPY data/ data/
COPY models/ models/
COPY utils/ utils/
COPY api_key.json api_key.json
COPY heartwise.config heartwise.config
COPY main.py main.py
COPY models_setup.py models_setup.py
COPY run_pipeline.bash run_pipeline.bash

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Define an optional build argument
ARG RUN_MODELS_SETUP=false

# Conditionally run models_setup.py based on the build argument
RUN if [ "$RUN_MODELS_SETUP" = "true" ]; \
    then echo "Running models_setup.py"; \
    python models_setup.py; \
    else echo "Skipping models_setup.py"; \
    fi

# Set entrypoint
ENTRYPOINT ["./run_pipeline.bash"]

# Use CMD to set the defualt mode 
CMD ["full_run"]