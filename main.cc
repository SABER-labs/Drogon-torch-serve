#include <drogon/drogon.h>

int main() {
    LOG_INFO << "Server running at 0.0.0.0:8088";
    drogon::app()
            .addListener("0.0.0.0", 8088)
            .loadConfigFile("../config.json")
            .run();
    return 0;
}
