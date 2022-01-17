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
for i in {0..8}; do curl "localhost:8088/classify" -F "image=@images/cat.jpg"; done # Run once to warmup.
wrk -t8 -c100 -d60 -s benchmark/upload.lua "http://localhost:8088/classify" --latency
```

## Benchmarking results
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

## Notes
* WIP: Just gets the job done for now, not production ready