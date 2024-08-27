
FROM ubuntu:24.04

WORKDIR /app
COPY . .

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/tim

RUN apt update
RUN apt install -y software-properties-common
RUN apt install -y wget
RUN apt install -y git
RUN apt install -y libfftw3-dev
RUN apt install -y g++
RUN apt install -y python3
RUN apt install -y python3-pip
RUN apt update

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
RUN apt update
RUN apt install -y cuda-11.8

RUN export CXX=$(which g++)
RUN export CUDACXX=$(which nvcc)
RUN git submodule update --init --recursive
RUN pip install .
