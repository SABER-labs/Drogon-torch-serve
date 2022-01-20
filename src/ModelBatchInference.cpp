//
// Created by vigi99 on 16/01/22.
//

#include "ModelBatchInference.h"

ModelBatchInference::ModelBatchInference() {
    Timer measure("ModelBatchInference constructor");
    std::ifstream i("../model_resources/class_names.json");
    i >> class_idx_to_names;
    if (torch::cuda::is_available()) {
        device_type = torch::kCUDA;
        model = torch::jit::load(std::filesystem::absolute("../model_resources/resnet18_traced_cuda.pt"), device_type);
        model.to(torch::kFloat16);
        LOG_INFO << "Model loaded onto CUDA";
    } else {
        device_type = torch::kCPU;
        model = torch::jit::load(std::filesystem::absolute("../model_resources/resnet18_traced_cpu.pt"), device_type);
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
            std::vector<std::reference_wrapper<std::promise<ModelResponse>>> response_promises;
            std::vector<torch::jit::IValue> inputs;
            for (size_t i = 0; i < tensors_to_process; ++i) {
                auto [res_promise, tensor_image_ref] = request_queue.front();
                auto tensor_image = tensor_image_ref.get();
                tensor_images.emplace_back(tensor_image);
                response_promises.emplace_back(res_promise);
                std::lock_guard<std::mutex> guard(request_queue_mutex);
                request_queue.pop();
            }

            auto batched_tensor = torch::cat(tensor_images, 0);

            if (device_type == torch::kCUDA) {
                batched_tensor = batched_tensor.to(device_type).permute({0, 3, 1, 2}).div(255).toType(torch::kFloat16);
            } else {
                batched_tensor = batched_tensor.to(device_type).permute({0, 3, 1, 2}).div(255);
            }

            inputs.emplace_back(batched_tensor);
            auto output = model.forward(inputs).toTensor();
            auto [confidences, values] = torch::softmax(output, 1).max(1);

            for (int64_t i = 0; i < output.size(0); ++i) {
                auto response_promise = response_promises[i];
                auto imagenet_class = class_idx_to_names[std::to_string(values[i].item<int>())];
                auto confidence = confidences[i].item<float>();
                auto response = ModelResponse{imagenet_class, confidence};
                response_promise.get().set_value(response);
            }

//            c10::cuda::CUDACachingAllocator::emptyCache();
        }

    }
}

ModelResponse ModelBatchInference::infer(const torch::Tensor &image_tensor) {
    // Add the image tensor to request queue
    // Timer measure("ModelBatchInference infer");
    std::promise<ModelResponse> response;
    auto response_future = response.get_future();
    {
        std::lock_guard<std::mutex> guard(request_queue_mutex);
        auto pair = std::pair(std::ref(response), std::ref(image_tensor));
        request_queue.push(pair);
    }

    auto model_response = response_future.get();
    // Send back results
    return std::move(model_response);
}
