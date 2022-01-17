# C++ Torch Server
### Serve torch models as rest-api using [Drogon](https://github.com/drogonframework/drogon), example included for resnet18 model for Imagenet. Benchmarks indicate around 20x performance improvement for resenet18 at peak load.

## Build & Run Instructions
```bash
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
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:8088
deactivate # Use after benchmarking is done and gunicorn is closed

cd ../.. # back to root folder
for i in {0..8}; do curl "localhost:8088/classify" -F "image=@images/cat.jpg"; done
wrk -t8 -c100 -d10 -s benchmark/fastapi_upload.lua "http://localhost:8088/classify" --latency
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
    Latency    62.94ms   23.30ms 128.98ms   59.62%
    Req/Sec   191.28     19.48   252.00     63.29%
  Latency Distribution
     50%   63.94ms
     75%   73.42ms
     90%  101.56ms
     99%  115.37ms
  91501 requests in 1.00m, 24.08MB read
Requests/sec:   1524.01
Transfer/sec:    410.77KB
```

`FastAPI + pytorch`
```bash
# OS: Ubuntu 21.10 x86_64
# Kernel: 5.15.14-xanmod1
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz
# GPU: NVIDIA GeForce RTX 3070
Running 20s test @ http://localhost:8088/classify
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   729.57ms  572.06ms   1.89s    49.40%
    Req/Sec    23.14     24.34   110.00     85.44%
  Latency Distribution
     50%  844.05ms
     75%    1.23s
     90%    1.41s
     99%    1.89s
  1476 requests in 20.03s, 307.02KB read
  Socket errors: connect 0, read 4, write 0, timeout 318
Requests/sec:     73.67
Transfer/sec:     15.32KB
```

## Dependencies
* [Libtorch](https://pytorch.org/get-started/locally/)
* [Torch Vision Library](https://github.com/pytorch/vision#using-the-models-on-c)
* Libopencv-dev `sudo apt-get install -y libopencv-dev`
* [Install Drogon](https://github.com/drogonframework/drogon/wiki/ENG-02-Installation)

## TODO
* ~~Will include multi-tenant batched inference on another thread as done in https://github.com/SABER-labs/torch_batcher~~
* ~~Use ThreadPool for batched inference.~~
* ~~FP16 Inference~~
* Int8 Inference using [FXGraph post-training quantization](https://pytorch.org/docs/stable/quantization.html)
* Resnet Int8 Quantization [example1](https://github.com/zanvari/resnet50-quantiztion/blob/main/quantization-resnet50.ipynb), [example2](https://github.com/SangbumChoi/PyTorch_Quantization/blob/9773c4397dbf6dd04c3e126524c36e398d8b60e6/quantization.py)
* Use [lockfree](https://theboostcpplibraries.com/boost.lockfree) queues

## Notes
* WIP: Just gets the job done for now, not production ready