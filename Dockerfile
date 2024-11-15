FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

WORKDIR /app
COPY . .

ENV TZ=Europe/London
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/tim

RUN apt update
RUN apt install -y software-properties-common
RUN apt install -y git
RUN apt install -y libfftw3-dev
RUN apt install -y g++
RUN apt install -y python3
RUN apt install -y python3-pip
RUN export CXX=$(which g++)
RUN export CUDACXX=$(which nvcc)
RUN git submodule update --init --recursive
RUN pip install -U pip
RUN pip install . 
