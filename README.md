# C++ Torch Server
### Serve torch models as rest-api using [Drogon](https://github.com/drogonframework/drogon), example included for resnet18 model for Imagenet.

## Build & Run Instructions
```bash
    mkdir build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    ./rest_server
```

## Dependencies
* libtorch
* libopencv
* Drogon