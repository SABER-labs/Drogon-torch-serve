//
// Created by vigi99 on 9/19/22.
//

#include "OnnxUtilities.h"

int64_t vectorProduct(const std::vector<int64_t> &v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
}

std::tuple<float, std::string> getTopResult(std::vector<float>::iterator begin, std::vector<float>::iterator end,
                                            const nlohmann::json &class_idx_to_names) {
    auto max = std::max_element(begin, end);
    auto max_idx = std::distance(begin, max);
    auto confidence = 1 / std::accumulate(begin, end, 0.0f, [max](float a, float b) { return a + std::exp(b - *max); });
    return std::make_tuple(confidence, class_idx_to_names[std::to_string(max_idx)]);
}

Ort::Session createOrtSession(const std::string &model_path) {
    Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "test");
    Ort::SessionOptions session_options;
    session_options.AddConfigEntry("session.set_denormal_as_zero", "1");
    session_options.DisableCpuMemArena();
    auto num_threads = std::thread::hardware_concurrency() >= 4 ? int(std::thread::hardware_concurrency() / 4) : 1;
    session_options.SetIntraOpNumThreads(num_threads);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), session_options);
    return session;
}

std::tuple<std::vector<float>, std::vector<float>, std::vector<int64_t>, std::vector<int64_t>>
generateInputOutputTensorValuesForORT(std::vector<cv::Mat> &images, int64_t num_classes) {
    auto batch_size = (int) images.size();
    auto input_tensor_size = images[0].size;
    std::vector<int64_t> inputDims = {batch_size, input_tensor_size[1], input_tensor_size[2], input_tensor_size[3]};
    std::vector<int64_t> outputDims = {batch_size, num_classes};
    size_t outputTensorSize = vectorProduct(outputDims);

    std::vector<float> inputTensorValues;
    std::vector<float> outputTensorValues(outputTensorSize);
    std::vector<Ort::Value> inputTensors;
    std::vector<Ort::Value> outputTensors;

    for (auto &image: images) {
        std::copy(image.begin<float>(), image.end<float>(), std::back_inserter(inputTensorValues));
    }

    return std::make_tuple(inputTensorValues, outputTensorValues, inputDims, outputDims);
}

cv::Mat processImage(const cv::Mat &image) {
    cv::Mat image_transformed, preprocessedImage;
    cv::resize(image, image_transformed,
               cv::Size(224, 224),
               cv::InterpolationFlags::INTER_CUBIC);
    cv::cvtColor(image_transformed, image_transformed,
                 cv::ColorConversionCodes::COLOR_BGR2RGB);
    image_transformed.convertTo(image_transformed, CV_32F, 1.0 / 255);

    cv::Mat channels[3];
    cv::split(image_transformed, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, image_transformed);
    cv::dnn::blobFromImage(image_transformed, preprocessedImage);
    return preprocessedImage;
}