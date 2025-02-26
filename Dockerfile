# Written by Joshua Guthrie (Department of Physics, University of Alberta, jdguthri@ualberta.ca), Feb 2025
# Docker file to build a container (with a bash shell) that has Python 3.9, curveball, and the dependency versions from 2021 required to run it
# The docker file should work to build the container on any OS - instructions for ubuntu-based linux distros are provided below

## Instructions
# To install docker from an ubuntu command line using apt: sudo apt install docker
# To build the container, navigate to the directory with this file and run: sudo docker build -t curveball_docker .
# To use (change path to the directory you want to have access to while working in the container): sudo docker run -v /path/to/work/directory/:/app/CurveBall -it curveball_docker   
# From there, the container will act like a normal linux terminal and curveball can be used as described in the documentation
# More details and instructions for installing docker, building from docker files, and using containers in other operating systems can be found at https://docs.docker.com/ 
 
FROM python:3.9

RUN apt-get update && apt-get install -y bash

WORKDIR /app

RUN pip install --no-cache-dir \
    curveball \
    future==0.18.2 \
    click==7.1.2 \
    lxml==4.6.3 \
    xlrd==1.2.0 \
    numpy==1.21.0 \
    scipy==1.7.0 \
    matplotlib==3.4.2 \
    pandas==1.3.0 \
    seaborn==0.11.1 \
    scikit-learn==0.24.2 \
    sympy==1.8 \
    lmfit==1.0.2 \
    webcolors==1.11

CMD ["/bin/bash"]
