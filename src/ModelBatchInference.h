//
// Created by vigi99 on 16/01/22.
//

#include <string>
#include <utility>
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
#include <coro/coro.hpp>

#pragma once

struct ModelResponse {
    std::string className;
    float confidence;

public:
    void setValues(std::string cn, float c) {
        className = std::move(cn);
        confidence = c;
    }
};

struct QueueRequest {
    std::reference_wrapper<ModelResponse> response;
    std::reference_wrapper<coro::event> e;
    std::reference_wrapper<const at::Tensor> image_tensor;
};

static const int MAX_WAIT_IN_MS = 4;
static const int MAX_BATCH_SIZE = 16;
static const int POLL_INTERVAL_MS = 1;
static constexpr int NUM_POOL_LOOPS = MAX_WAIT_IN_MS / POLL_INTERVAL_MS;

class ModelBatchInference {
public:
    ModelBatchInference();

    [[noreturn]] void foreverBatchInfer();

    drogon::Task<ModelResponse> infer(const torch::Tensor &);

private:
    std::queue<QueueRequest> request_queue;
    std::mutex request_queue_mutex;
    torch::DeviceType device_type;

    torch::jit::Module model;
    nlohmann::json class_idx_to_names;
};