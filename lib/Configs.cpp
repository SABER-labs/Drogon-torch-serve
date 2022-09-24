#include "Configs.h"

int Configs::getEnvVariable(const char* env_var_name, int default_value) {
    char * val = getenv(env_var_name);
    int num_inference_engines = val == nullptr ? default_value : std::stoi(val);
    return num_inference_engines;
}

const int Configs::MAX_WAIT_IN_MS = 2;
const int Configs::POLL_INTERVAL_MS = 1;
const int Configs::MAX_BATCH_SIZE = 32;
const int Configs::NUM_POOL_LOOPS = MAX_WAIT_IN_MS / POLL_INTERVAL_MS;

// Configurations for controlling the number of inference threads
const int Configs::NUM_CONTROLLER_THREADS = Configs::getEnvVariable("NUM_CONTROLLER_THREADS", std::max(1, int(std::thread::hardware_concurrency() * 0.2)));
const int Configs::NUM_INFERENCE_THREADS = Configs::getEnvVariable("NUM_INFERENCE_THREADS", std::max(1, int(std::thread::hardware_concurrency() * 0.2)));

// Static configurations for the model
const int Configs::MODEL_THREADS_PER_SESSION =  Configs::getEnvVariable("MODEL_THREADS_PER_SESSION", std::max(1, ((int) std::thread::hardware_concurrency() - NUM_CONTROLLER_THREADS - 1) /  NUM_INFERENCE_THREADS));

