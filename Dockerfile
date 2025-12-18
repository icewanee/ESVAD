FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update
RUN apt-get install -y software-properties-common git screen nano htop curl unzip wget sudo build-essential
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.11 python3.11-distutils python3-pip openjdk-21-jdk

# CPG
WORKDIR /app
RUN git clone https://github.com/icewanee/cpg.git
WORKDIR /app/cpg
RUN mv gradle.properties.example gradle.properties
RUN ./gradlew installDist && cp -r cpg-neo4j/build/install/cpg-neo4j /app/cpg-neo4j && rm -rf ~/.gradle/caches && rm -rf ../cpg

# Python dep
WORKDIR /app
COPY requirements.txt requirements.txt
RUN apt-get install python3.11-dev -y
RUN python3.11 -m pip install --upgrade pip setuptools wheel && python3.11 -m pip install -r requirements.txt && python3.11 -m pip cache purge

# LLVM 15
RUN python3 -m pip install pyparsing six
RUN apt-get update && apt-get install -y wget gnupg && \
    mkdir -p /etc/dpkg/dpkg.cfg.d && \
    echo "force-overwrite" > /etc/dpkg/dpkg.cfg.d/force-overwrite && \
    wget https://apt.llvm.org/llvm.sh && chmod +x llvm.sh && ./llvm.sh 15 && \
    rm llvm.sh
RUN apt-get install -y mingw-w64-common mingw-w64-x86-64-dev g++-mingw-w64-x86-64

RUN python3 -m pip install gdown
COPY . /app

#CMD ["/bin/bash"]
CMD ["sleep","infinity"]
