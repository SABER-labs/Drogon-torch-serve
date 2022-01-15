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
    Latency   185.64ms   88.26ms 455.58ms   72.36%
    Req/Sec    64.03     19.82   181.00     70.27%
  Latency Distribution
     50%  170.45ms
     75%  237.23ms
     90%  334.73ms
     99%  398.80ms
  5146 requests in 10.08s, 1.35MB read
Requests/sec:    510.34
Transfer/sec:    137.55KB
```

## Dependencies
* libtorch
* libopencv
* Drogon