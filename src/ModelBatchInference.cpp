//
// Created by vigi99 on 16/01/22.
//

#include "ModelBatchInference.h"

ModelBatchInference::ModelBatchInference() {
    Timer measure("ModelBatchInference constructor");
    std::ifstream i("../model_resources/class_names.json");
    i >> class_idx_to_names;
    if (torch::cuda::is_available()) {
        model = torch::jit::load(std::filesystem::absolute("../model_resources/resnet18_traced_cuda.pt"), torch::kCUDA);
        model.to(torch::kFloat16);
        LOG_INFO << "Model loaded onto CUDA";
    } else {
        model = torch::jit::load(std::filesystem::absolute("../model_resources/resnet18_traced_cpu.pt"), torch::kCPU);
        LOG_INFO << "Model loaded onto CPU";
    }

    {
        torch::NoGradGuard no_grad_t;
        model.eval();
    }

}

void ModelBatchInference::foreverBatchInfer() {
    for (;;) {
        for (size_t i = 0; i < NUM_POOL_LOOPS; ++i) {
            if (request_queue.size() < MAX_BATCH_SIZE) {
                std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
            } else {
                break;
            }
        }

        if (!request_queue.empty()) {
            c10::InferenceMode no_grad(true);
            int tensors_to_process = std::min((int) request_queue.size(), MAX_BATCH_SIZE);
            std::vector<torch::Tensor> tensor_images;
            std::vector<std::string> request_ids;
            std::vector<torch::jit::IValue> inputs;
            for (size_t i = 0; i < tensors_to_process; ++i) {
                auto [req_id, tensor_image_ref] = request_queue.front();
                auto tensor_image = tensor_image_ref.get();
                tensor_images.emplace_back(tensor_image);
                request_ids.emplace_back(req_id);
                std::lock_guard<std::mutex> guard(request_queue_mutex);
                request_queue.pop();
            }

            auto batched_tensor = torch::cat(tensor_images, 0)
                    .to(torch::kCUDA)
                    .permute({0, 3, 1, 2})
                    .toType(torch::kFloat16)
                    .div(255);
            inputs.emplace_back(batched_tensor);
            auto output = model.forward(inputs).toTensor();
            auto [confidences, values] = torch::softmax(output, 1).max(1);

            for (int64_t i = 0; i < output.size(0); ++i) {
                auto response_id = request_ids[i];
                auto imagenet_class = class_idx_to_names[std::to_string(values[i].item<int>())];
                auto confidence = confidences[i].item<float>();
                auto response = ModelResponse{imagenet_class, confidence};
                std::lock_guard<std::mutex> guard(response_queue_mutex);
                response_queue.insert({response_id, response});
            }

//            c10::cuda::CUDACachingAllocator::emptyCache();
        }

    }
}

ModelResponse ModelBatchInference::infer(const std::string &req_id, const torch::Tensor &image_tensor) {
    // Add the image tensor to request queue
    // Timer measure("ModelBatchInference infer");
    {
        std::lock_guard<std::mutex> guard(request_queue_mutex);
        auto pair = std::pair(req_id, std::ref(image_tensor));
        request_queue.push(pair);
    }

    // Wait for the image to turn up in the response queue.
    while (response_queue.find(req_id) == response_queue.end()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    }

    // Remove from response queue
    auto model_response = response_queue[req_id];

    {
        std::lock_guard<std::mutex> guard(response_queue_mutex);
        response_queue.erase(req_id);
    }

    // Send back results
    return std::move(model_response);
}
