#include <drogon/drogon.h>
#include "lib/Configs.h"

int main() {
    LOG_INFO << "Server running at 0.0.0.0:8088";
    LOG_INFO << "NUM_CONTROLLER_THREADS: " << Configs::NUM_CONTROLLER_THREADS;
    drogon::app()
            .addListener("0.0.0.0", 8088)
            .loadConfigFile("/app/config.json")
            .setThreadNum(Configs::NUM_CONTROLLER_THREADS)
            .run();
    return 0;
}
