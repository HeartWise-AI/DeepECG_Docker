# Pillow 13 security fixes require Python >= 3.11.
FROM python:3.11

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

# Install CUDA PyTorch first, then resolve the remaining dependencies from PyPI.
# The PyTorch index also mirrors Pillow, but does not carry its security release.
RUN uv pip install --system --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 && \
    uv pip install --system -r requirements.txt

# Pin the upstream Python 3.11 compatibility fixes. Its setup.py still declares
# the old Hydra/OmegaConf bounds; use the compatible versions for this image.
RUN git clone https://github.com/HeartWise-AI/fairseq-signals && \
    cd fairseq-signals && \
    git checkout 980e705ce957c6f46b62f0ca2cfa2d846e112bae && \
    sed -i 's/"hydra-core<1.1"/"hydra-core==1.3.2"/; s/"omegaconf<2.1"/"omegaconf==2.3.1"/' setup.py && \
    uv pip install --system --editable ./    

# Copy the rest of the application code into the container
COPY data/ data/
COPY ingestion/ ingestion/
COPY models/ models/
COPY utils/ utils/
COPY thresholds/ thresholds/
COPY heartwise.config heartwise.config
COPY main.py main.py
COPY models_setup.py models_setup.py
COPY run_pipeline.bash run_pipeline.bash

# Create an empty credential file; real keys are supplied only through secret mounts.
RUN python -m utils.credentials

# Make the bash script executable
RUN chmod +x run_pipeline.bash

# Define an optional build argument
ARG RUN_MODELS_SETUP=false

# Conditionally run models_setup.py based on the build argument
RUN --mount=type=secret,id=hf_api_key,target=/app/api_key.json \
    if [ "$RUN_MODELS_SETUP" = "true" ]; \
    then echo "Running models_setup.py"; \
    python models_setup.py; \
    else echo "Skipping models_setup.py"; \
    fi

CMD ["tail", "-f", "/dev/null"]
