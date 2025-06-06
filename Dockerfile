FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies, including OpenGL runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY ./app/requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

# Initialize and update git submodules
COPY .git .git
COPY .gitmodules .gitmodules
RUN git submodule update --init --recursive

# Copy application files
COPY ./app /app
COPY ./input /app/input
COPY ./entrypoint.sh /entrypoint.sh

RUN chmod +x /entrypoint.sh

# Set entrypoint
ENTRYPOINT ["sh", "/entrypoint.sh"]
