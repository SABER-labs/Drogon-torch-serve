//
// Created by vigi99 on 16/01/22.
//
#pragma once
#include <fstream>
#include <string>
#include <utility>
#include <vector>
#include <queue>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <drogon/drogon.h>
#include <coro/coro.hpp>
#include "includes/json.hpp"
#include "lib/Utilities.hpp"
#include "lib/OnnxUtilities.h"
#include "lib/Configs.h"

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
    std::reference_wrapper<const cv::Mat> image_tensor;
};

class ModelBatchInference {
public:
    ModelBatchInference();

    [[noreturn]] void foreverBatchInfer();

    drogon::Task<ModelResponse> infer(const cv::Mat &);

private:
    std::queue<QueueRequest> request_queue;
    std::mutex request_queue_mutex;
    Ort::Env env{nullptr};
    Ort::Session session{nullptr};
    nlohmann::json class_idx_to_names;
};