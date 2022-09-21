//
// Created by vigi99 on 16/01/22.
//

#include <fstream>
#include "ModelBatchInference.h"

ModelBatchInference::ModelBatchInference() {
    Timer measure("ModelBatchInference constructor");
    std::ifstream class_names_path("/app/model_resources/class_names.json");
    class_idx_to_names = nlohmann::json::parse(class_names_path);
    session = createOrtSession("/app/model_resources/resnet18-v2-7.onnx");
    LOG_INFO << "Model loaded onto CPU";
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
            int tensors_to_process = std::min((int) request_queue.size(), MAX_BATCH_SIZE);
            std::vector<std::reference_wrapper<const cv::Mat>> tensor_images;
            std::vector<std::reference_wrapper<coro::event>> response_events;
            std::vector<std::reference_wrapper<ModelResponse>> responses;
            for (int i = 0; i < tensors_to_process; ++i) {
                auto [model_response, e, tensor_image] = request_queue.front();
                tensor_images.push_back(tensor_image);
                response_events.push_back(e);
                responses.push_back(model_response);
                std::lock_guard<std::mutex> guard(request_queue_mutex);
                request_queue.pop();
            }

            auto num_classes = (int) class_idx_to_names.size();
            auto [inputTensorValues, outputTensorValues,
                    inputDims, outputDims] = generateInputOutputTensorValuesForORT(tensor_images, num_classes);

            Ort::AllocatorWithDefaultOptions allocator;
            char* inputName = session.GetInputName(0, allocator);
            char* outputName = session.GetOutputName(0, allocator);
            std::vector<char *> inputNames{inputName};
            std::vector<char*> outputNames{outputName};

            auto memoryInfo = Ort::MemoryInfo::CreateCpu(
                    OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeDefault);
            std::vector<Ort::Value> inputTensors;
            std::vector<Ort::Value> outputTensors;

            inputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, inputTensorValues.data(), inputTensorValues.size(), inputDims.data(),
                    inputDims.size()));
            outputTensors.push_back(Ort::Value::CreateTensor<float>(
                    memoryInfo, outputTensorValues.data(), outputTensorValues.size(),
                    outputDims.data(), outputDims.size()));

            session.Run(Ort::RunOptions{nullptr}, inputNames.data(), inputTensors.data(),
                        inputTensors.size(), outputNames.data(), outputTensors.data(),
                        outputTensors.size());

            allocator.Free(inputName);
            allocator.Free(outputName);

            for (int64_t i = 0; i < tensors_to_process; ++i) {
                auto response = responses[i];
                auto e = response_events[i];
                auto begin_idx = outputTensorValues.begin() + i * num_classes;
                auto end_idx = outputTensorValues.begin() + (i + 1) * num_classes;
                auto [confidence, imagenet_class] = getTopResult(begin_idx, end_idx, class_idx_to_names);
                response.get().setValues(imagenet_class, confidence);
                e.get().set();
            }

        }

    }
}

drogon::Task<ModelResponse> ModelBatchInference::infer(const cv::Mat& image_tensor) {
    // Add the image tensor to request queue
    Timer measure("ModelBatchInference infer");
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
