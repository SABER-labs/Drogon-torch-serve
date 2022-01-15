#include <drogon/drogon.h>
#include <torch/torch.h>
int main() {
    LOG_INFO << "Server running at 127.0.0.1:8088";
    at::globalContext().setBenchmarkCuDNN(true);
    drogon::app()
    .addListener("0.0.0.0",8088)
    .loadConfigFile("../config.json")
    .run();
    return 0;
}
