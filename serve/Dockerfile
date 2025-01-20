FROM nvidia/cuda:11.2.0-runtime-ubuntu20.04
LABEL maintainer="Akinola Antony Wilson <akinolawilson@gmail.com>"

# Allow passing in decision threshold and model version during build of serving container.
ARG DECISION_THRESHOLD=0.5
ARG MODEL_VERSION="docker-env-model-version"
# Setting decision_threshold and model_version and env vars 
ENV DECISION_THRESHOLD=${DECISION_THRESHOLD}
ENV MODEL_VERSION=${MODEL_VERSION}
# install utilities
RUN apt-get update && \
    apt-get install --no-install-recommends -y curl
# audio processing dependencies 
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y libsndfile-dev
# Install python 
RUN apt-get install -y python3
RUN apt-get install -y python3-pip
# Installing python dependencies
RUN python3 -m pip --no-cache-dir install --upgrade pip && \
    python3 --version && \
    pip3 --version

COPY ./requirements.txt .
RUN pip3 --timeout=300 --no-cache-dir install -r requirements.txt

# Copy model files
COPY ./model /model

# Copy app files
COPY ./app /app

COPY ./start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 80
CMD ["/bin/bash"]
