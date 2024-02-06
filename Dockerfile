#############################################
# Select the OS
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

USER root

ENV TZ=US/Eastern
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt update && apt install -y tzdata

#############################################
# Install dependencies
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.9 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.9 -m pip install --upgrade pip 

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install --no-cache-dir  tensorflow==2.10 \
    nilearn pandas \
    scipy numpy matplotlib tqdm imageio scikit-image \
    scikit-learn itk-elastix SimpleITK nibabel intensity-normalization[ants]\
    wandb jupyter opencv-python  

RUN python3.9 -c "import tensorflow"

#############################################
# Setup default ENV_FOLDER/v0 directory
ENV ENV_FOLDER="/ENV_FOLDER/v0"
RUN mkdir -p ${ENV_FOLDER}
WORKDIR ${ENV_FOLDER}
COPY *.py ${ENV_FOLDER}/
COPY *.csv ${ENV_FOLDER}/
COPY scripts ${ENV_FOLDER}/scripts
COPY golden_image ${ENV_FOLDER}/golden_image
COPY model_weights ${ENV_FOLDER}/model_weights
COPY preprocessing ${ENV_FOLDER}/preprocessing
# reload bashrc
RUN ["/bin/bash", "-c", "source /root/.bashrc"]

#############################################
# Configure entrypoint
RUN chmod a+x /ENV_FOLDER/v0/run
CMD [ "python3.9", "./main.py" ]
