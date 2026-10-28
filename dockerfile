# syntax=docker/dockerfile:1
FROM python:3.10

# Set the working directory
WORKDIR /app

# Install git and other necessary packages
RUN apt-get update && \
    apt-get install -y git build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv
    
# Copy the files into the container
COPY requirements.txt .

# Install required libraries using uv
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements.txt

# Clone the fairseq-signals repository and install it
RUN git clone https://github.com/HeartWise-AI/fairseq-signals && \
    cd fairseq-signals && \
    uv pip install --system --editable ./    

# Copy the rest of the application code into the container
COPY data/ data/
COPY models/ models/
COPY utils/ utils/
COPY thresholds/ thresholds/
COPY clsa_cvd/ clsa_cvd/
COPY heartwise.config heartwise.config
COPY main.py main.py
COPY models_setup.py models_setup.py
COPY run_pipeline.bash run_pipeline.bash

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Define an optional build argument
ARG RUN_MODELS_SETUP=false

# Conditionally run models_setup.py based on the build argument
RUN --mount=type=secret,id=hf_api_key,target=/app/api_key.json \
    if [ "$RUN_MODELS_SETUP" = "true" ]; \
    then echo "Running models_setup.py"; \
    test -r /app/api_key.json || { echo "RUN_MODELS_SETUP=true requires --secret id=hf_api_key,src=PATH_TO_KEY_JSON" >&2; exit 1; }; \
    python models_setup.py; \
    else echo "Skipping models_setup.py"; \
    fi

CMD ["tail", "-f", "/dev/null"]
