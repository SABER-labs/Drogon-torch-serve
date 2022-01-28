# C++ Torch Server
### Serve torch models as rest-api using [Drogon](https://github.com/drogonframework/drogon), example included for resnet18 model for Imagenet. Benchmarks show improvement of ~6-10x throughput and latencies for resnet18 at peak load.

## Build & Run Instructions
```bash
# Create Optimized models for your machine.
python3 optimize_model_for_inference.py

# Build and Run Server
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
./rest_server
```

## Client Instructions
```bash
curl "localhost:8088/classify" -F "image=@images/cat.jpg"
```

## Benchmarking Instructions
```bash
# Drogon + libtorch
for i in {0..8}; do curl "localhost:8088/classify" -F "image=@images/cat.jpg"; done # Run once to warmup.
wrk -t8 -c100 -d60 -s benchmark/upload.lua "http://localhost:8088/classify" --latency
```

```bash
# FastAPI + pytorch
cd benchmark/python_fastapi
python3 -m venv env
source env/bin/activate
python3 -m pip install -r requirements.txt # Run just once to isntall dependencies to folder.
gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1: # Best performance on my machine, tried 3/4 also.
deactivate # Use after benchmarking is done and gunicorn is closed

cd ../.. # back to root folder
for i in {0..8}; do curl "localhost:8088/classify" -F "image=@images/cat.jpg"; done
wrk -t8 -c100 -d60 -s benchmark/fastapi_upload.lua "http://localhost:8088/classify" --latency
```

## Benchmarking results
`Drogon + libtorch`
```bash
# OS: Ubuntu 21.10 x86_64
# Kernel: 5.15.14-xanmod1
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz
# GPU: NVIDIA GeForce RTX 3070
Running 1m test @ http://localhost:8088/classify
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    39.30ms   10.96ms  95.51ms   70.50%
    Req/Sec   306.58     28.78   390.00     70.92%
  Latency Distribution
     50%   37.40ms
     75%   45.69ms
     90%   54.57ms
     99%   69.34ms
  146612 requests in 1.00m, 30.34MB read
Requests/sec:   2441.60
Transfer/sec:    517.41KB
```

`FastAPI + pytorch`
```bash
# OS: Ubuntu 21.10 x86_64
# Kernel: 5.15.14-xanmod1
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz
# GPU: NVIDIA GeForce RTX 3070
Running 1m test @ http://localhost:8088/classify
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   449.50ms  239.30ms   1.64s    70.39%
    Req/Sec    33.97     26.41   121.00     83.46%
  Latency Distribution
     50%  454.64ms
     75%  570.73ms
     90%  743.54ms
     99%    1.16s
  12981 requests in 1.00m, 2.64MB read
Requests/sec:    216.13
Transfer/sec:     44.96KB
```

## Architecture
* API request handing and model Pre-processing in the Drogon Controller `controllers/ImageClass.cc`
* Batched Model Inference logic & post-processing in `src/ModelBatchInference.cpp`
* TODO: Currently pre-processing, inference & post-processing isn't seperated out.

## Dependencies
* [Libtorch](https://pytorch.org/get-started/locally/)
* [Torch Vision Library](https://github.com/pytorch/vision#using-the-models-on-c)
* Libopencv-dev `sudo apt-get install -y libopencv-dev`
* [Install Drogon](https://github.com/drogonframework/drogon/wiki/ENG-02-Installation)

## TODO
* ~~Will include multi-tenant batched inference on another thread as done in https://github.com/SABER-labs/torch_batcher~~
* ~~Use ThreadPool for batched inference.~~
* ~~FP16 Inference~~
* ~~Uses c++20 coroutines for wait free event loop tasks.~~
* Int8 Inference using [FXGraph post-training quantization](https://pytorch.org/docs/stable/quantization.html)
* Resnet Int8 Quantization [example1](https://github.com/zanvari/resnet50-quantiztion/blob/main/quantization-resnet50.ipynb), [example2](https://github.com/SangbumChoi/PyTorch_Quantization/blob/9773c4397dbf6dd04c3e126524c36e398d8b60e6/quantization.py)
* Use [lockfree](https://theboostcpplibraries.com/boost.lockfree) queues
* Dockerize for easy usage.
* Seperate Pre-Process, Infer and post-preprocessing.

## Notes
* WIP: Just gets the job done for now, not production ready, though tested regularly.