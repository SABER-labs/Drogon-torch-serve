# C++ Torch Server
### Serve torch models as rest-api using [Drogon](https://github.com/drogonframework/drogon), example included for resnet18 model for Imagenet.

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
curl "localhost:8088/classify" -F "image=@images/cat.jpg" # Run once to warmup.
wrk -t8 -c100 -d10 -s benchmark/upload.lua "http://localhost:8088/classify" --latency
```

## Benchmarking results
```bash
# OS: Ubuntu 21.10 x86_64
# Kernel: 5.15.14-xanmod1
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz
# GPU: NVIDIA GeForce RTX 3070
Running 10s test @ http://localhost:8088/classify
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency   133.73ms   55.97ms 293.76ms   60.80%
    Req/Sec    89.21     22.13   202.00     73.63%
  Latency Distribution
     50%  116.53ms
     75%  190.91ms
     90%  205.15ms
     99%  259.52ms
  7159 requests in 10.09s, 1.88MB read
Requests/sec:    709.62
Transfer/sec:    191.27KB
```

## Dependencies
* libtorch
* libopencv
* Drogon

## Notes
* WIP: Just gets the job done for now, not production ready
* Will include multi-tenant batched inference on another thread as done in https://github.com/SABER-labs/torch_batcher