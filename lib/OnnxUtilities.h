//
// Created by vigi99 on 9/19/22.
//
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "includes/json.hpp"
#include <trantor/utils/Logger.h>
#include <thread>

int64_t vectorProduct(const std::vector<int64_t>&);

std::tuple<float, std::string> getTopResult(const float*, const float*, const nlohmann::json&);

Ort::Session createOrtSession(Ort::Env&, const std::string&);

std::tuple<std::vector<float>, std::vector<int64_t>> generateInputOutputTensorValuesForORT(std::vector<std::reference_wrapper<const cv::Mat>>&);

cv::Mat processImage(std::reference_wrapper<const cv::Mat>);

uint getNumInferenceEngineThreads();

