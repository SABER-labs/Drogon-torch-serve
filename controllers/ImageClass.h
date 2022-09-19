#pragma once

#include <fmt/core.h>
#include <drogon/HttpController.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "lib/ModelBatchInference.h"

using namespace drogon;

class ImageClass : public drogon::HttpController<ImageClass> {
public:
    METHOD_LIST_BEGIN
        ADD_METHOD_TO(ImageClass::classify, "/classify", Post);
    METHOD_LIST_END

    ImageClass();

    const static int NUM_INFERENCE_ENGINES = 4;
private:
    std::vector<std::unique_ptr<ModelBatchInference>> batch_inference_engines;
protected:
    Task<> classify(HttpRequestPtr req, std::function<void(const HttpResponsePtr &)> callback);
};
