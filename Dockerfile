FROM drogonframework/drogon:latest

# Install dependency: OpenCV
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get install -y \
    libopencv-dev \
    tar wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install dependency: ONNX-Runtime
WORKDIR /
RUN mkdir /onnxruntime && cd /onnxruntime \
    && wget https://github.com/microsoft/onnxruntime/releases/download/v1.12.1/onnxruntime-linux-x64-1.12.1.tgz \
    && tar -xvf onnxruntime-linux-x64-1.12.1.tgz \
    && rm onnxruntime-linux-x64-1.12.1.tgz

# Install dependency: development-modules
RUN DEBIAN_FRONTEND="noninteractive" apt-get update && apt-get install -y \
    build-essential gcc g++ gdb make ninja-build cmake autoconf automake \
    libtool pkg-config \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

WORKDIR /app
RUN mkdir cmake-build-release && cd cmake-build-release \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. \
    && ninja

# Run
WORKDIR /app/cmake-build-release
EXPOSE 8088
CMD ["./blaze"]