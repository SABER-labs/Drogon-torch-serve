//
// Created by vigi99 on 16/01/22.
//

#include <string>
#include <vector>
#include <queue>
#include <chrono>
#include <thread>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <torchvision/models/resnet.h>
#include <torch/expanding_array.h>
#include "includes/json.hpp"
#include "src/Utilities.h"
#include <drogon/drogon.h>

#pragma once

struct ModelResponse {
    std::string className;
    float confidence;
};

static const int MAX_WAIT_IN_MS = 8;
static const int MAX_BATCH_SIZE = 32;
static const int POLL_INTERVAL_MS = 2;
static constexpr int NUM_POOL_LOOPS = MAX_WAIT_IN_MS / POLL_INTERVAL_MS;

class ModelBatchInference {
public:
    ModelBatchInference();

    [[noreturn]] void foreverBatchInfer();

    ModelResponse infer(const std::string &req_id, const torch::Tensor &);

private:
    std::queue<std::pair<std::string, std::reference_wrapper<const at::Tensor>>> request_queue;
    std::unordered_map<std::string, ModelResponse> response_queue;
    std::mutex response_queue_mutex;
    std::mutex request_queue_mutex;

    torch::jit::Module model;
    nlohmann::json class_idx_to_names;
};