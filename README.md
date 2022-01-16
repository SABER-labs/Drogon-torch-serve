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
wrk -t8 -c100 -d20 -s benchmark/upload.lua "http://localhost:8088/classify" --latency
```

## Benchmarking results
```bash
# OS: Ubuntu 21.10 x86_64
# Kernel: 5.15.14-xanmod1
# CPU: AMD Ryzen 9 5900X (24) @ 3.700GHz
# GPU: NVIDIA GeForce RTX 3070
Running 20s test @ http://localhost:8088/classify
  8 threads and 100 connections
  Thread Stats   Avg      Stdev     Max   +/- Stdev
    Latency    63.92ms   31.09ms 150.93ms   71.47%
    Req/Sec   188.43     27.10   260.00     65.65%
  Latency Distribution
     50%   55.00ms
     75%   74.07ms
     90%  121.52ms
     99%  135.23ms
  30118 requests in 20.08s, 7.93MB read
Requests/sec:   1499.64
Transfer/sec:    404.20KB
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

## Notes
* WIP: Just gets the job done for now, not production ready