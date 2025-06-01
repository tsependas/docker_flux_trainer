FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime

# Declare build-time arg, then expose it as an ENV inside the image
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

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

RUN python download_model.py

# Set entrypoint
#CMD ["python3", "-u", "start.py"]
CMD ["/bin/bash", "start.sh"]
