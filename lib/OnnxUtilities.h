//
// Created by vigi99 on 9/19/22.
//
#pragma once

#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include "includes/json.hpp"

int64_t vectorProduct(const std::vector<int64_t>&);

std::tuple<float, std::string> getTopResult(std::vector<float>::iterator, std::vector<float>::iterator, const nlohmann::json&);

Ort::Session createOrtSession(const std::string&);

std::tuple<std::vector<float>, std::vector<float>, std::vector<int64_t>, std::vector<int64_t>> generateInputOutputTensorValuesForORT(std::vector<cv::Mat>&, int64_t);

cv::Mat processImage(const cv::Mat&);