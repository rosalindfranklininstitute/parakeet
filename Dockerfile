FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /app
COPY . .

RUN apt update
RUN apt install -y git
RUN apt install -y libfftw3-dev
RUN apt install -y g++
RUN apt install -y python3
RUN apt install -y python3-pip
RUN export CXX=$(which g++)
RUN export CUDACXX=$(which nvcc)
RUN git submodule update --init --recursive
RUN pip install .

