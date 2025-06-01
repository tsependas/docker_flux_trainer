FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Install dependencies, including OpenGL runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

COPY ./app/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

RUN git clone https://github.com/ostris/ai-toolkit.git /app/ai-toolkit

# Copy application files
COPY ./app /app

# RUN python download_model.py
RUN chmod +x entrypoint.sh

# Set entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]
