FROM drogonframework/drogon:latest

# Install dependency: OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    gcc g++ make ninja-build cmake \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY ./ /app

WORKDIR /app
RUN mkdir -p build && cd build \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release .. \
    && ninja

WORKDIR /
RUN wget https://github.com/microsoft/mimalloc/archive/refs/tags/v2.0.6.tar.gz \
    && tar xvzf v2.0.6.tar.gz \
    && rm -rf v2.0.6.tar.gz \
    && cd mimalloc-2.0.6 \
    && mkdir -p out/release && cd out/release \
    && cmake -G Ninja -DCMAKE_BUILD_TYPE=Release ../.. \
    && ninja \
    && ninja install

# Run
WORKDIR /app/build
ENV LD_PRELOAD=/usr/local/lib/libmimalloc.so
ENTRYPOINT ["./blaze"]
