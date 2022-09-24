//
// Created by vigi99 on 9/19/22.
//

#include "OnnxUtilities.h"

int64_t vectorProduct(const std::vector<int64_t> &v) {
    return std::accumulate(v.begin(), v.end(), 1, std::multiplies<>());
}

std::tuple<float, std::string> getTopResult(const float *begin, const float *end,
                                            const nlohmann::json &class_idx_to_names) {
    auto max = std::max_element(begin, end);
    auto max_idx = std::distance(begin, max);
    auto confidence = 1 / std::accumulate(begin, end, 0.0f, [max](float a, float b) { return a + std::exp(b - *max); });
    return std::make_tuple(confidence, class_idx_to_names[std::to_string(max_idx)]);
}

Ort::Session createOrtSession(Ort::Env &env, const std::string &model_path) {
    Ort::SessionOptions session_options;
    int num_threads;

    #if USE_GPU == 0
        LOG_INFO << "GPU is not available, using CPU execution provider";
        session_options.AddConfigEntry("session.set_denormal_as_zero", "1");
        session_options.DisableCpuMemArena();
        num_threads = Configs::MODEL_THREADS_PER_SESSION;
    #elif USE_GPU == 1
        LOG_INFO << "GPU is available, using CUDA execution provider";
        OrtCUDAProviderOptions cudaOption;
        session_options.AppendExecutionProvider_CUDA(cudaOption);
        num_threads = 1;
    #endif

    session_options.SetIntraOpNumThreads(num_threads);
    LOG_INFO << "Setting intra_op threads to " << num_threads << " .";

    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, model_path.c_str(), session_options);
    LOG_INFO << "Loaded model from " << model_path;

    return session;
}

std::tuple<std::vector<float>, std::vector<int64_t>>
generateInputOutputTensorValuesForORT(std::vector<std::reference_wrapper<const cv::Mat>> &images) {
    auto batch_size = (int) images.size();

    std::vector<cv::Mat> processed_images;

    for (auto &image : images) {
        processed_images.emplace_back(processImage(image));
    }

    auto input_tensor_size = processed_images[0].size;
    std::vector<int64_t> inputDims = {batch_size, input_tensor_size[1], input_tensor_size[2], input_tensor_size[3]};

    std::vector<float> inputTensorValues;
    std::vector<Ort::Value> inputTensors;

    for (auto &image: processed_images) {
        std::copy(image.begin<float>(), image.end<float>(), std::back_inserter(inputTensorValues));
    }

    return std::make_tuple(inputTensorValues, inputDims);
}

cv::Mat processImage(std::reference_wrapper<const cv::Mat> image) {
    cv::Mat image_transformed, preprocessedImage;
    cv::resize(image.get(), image_transformed,
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

