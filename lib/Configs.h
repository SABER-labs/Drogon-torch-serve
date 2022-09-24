#pragma once

#include <thread>
#include <cstdlib>

// Language: cpp

class Configs {
public:
    // Configurations for controlling per inference thread batch size and max wait time
    static const int MAX_WAIT_IN_MS;
    static const int MAX_BATCH_SIZE;
    static const int POLL_INTERVAL_MS;
    static const int NUM_POOL_LOOPS;

    // Configurations for controlling the number of inference threads
    static const int NUM_CONTROLLER_THREADS;
    static const int NUM_INFERENCE_THREADS;

    // Static configurations for the model
    static const int MODEL_THREADS_PER_SESSION;

private:
    static int getEnvVariable(const char*, int);
};

