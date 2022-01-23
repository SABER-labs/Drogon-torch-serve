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
            std::vector<std::reference_wrapper<coro::event>> response_events;
            std::vector<std::reference_wrapper<ModelResponse>> responses;
            std::vector<torch::jit::IValue> inputs;
            for (int i = 0; i < tensors_to_process; ++i) {
                auto [model_response, e, tensor_image] = request_queue.front();
                tensor_images.push_back(tensor_image.get());
                response_events.push_back(e);
                responses.push_back(model_response);
                std::lock_guard<std::mutex> guard(request_queue_mutex);
                request_queue.pop();
            }

            auto batched_tensor = torch::cat(tensor_images, 0);

            if (device_type == torch::kCUDA) {
                batched_tensor = batched_tensor.to(device_type).toType(torch::kFloat16).permute({0, 3, 1, 2}).div(255);
            } else {
                batched_tensor = batched_tensor.to(device_type).permute({0, 3, 1, 2}).div(255);
            }

            inputs.emplace_back(batched_tensor);
            auto output = model.forward(inputs).toTensor();
            auto [confidences, values] = torch::softmax(output, 1).max(1);

            for (int64_t i = 0; i < output.size(0); ++i) {
                auto response = responses[i];
                auto e = response_events[i];
                auto imagenet_class = class_idx_to_names[std::to_string(values[i].item<int>())];
                auto confidence = confidences[i].item<float>();
                response.get().setValues(imagenet_class, confidence);
                e.get().set();
            }

//            c10::cuda::CUDACachingAllocator::emptyCache();
        }

    }
}

drogon::Task<ModelResponse> ModelBatchInference::infer(const torch::Tensor &image_tensor) {
    // Add the image tensor to request queue
    // Timer measure("ModelBatchInference infer");
    ModelResponse model_response;
    coro::event e;
    {
        std::lock_guard<std::mutex> guard(request_queue_mutex);
        request_queue.emplace(std::ref(model_response), std::ref(e), std::ref(image_tensor));
    }

    co_await e;
    // Send back results
    co_return std::move(model_response);
}
