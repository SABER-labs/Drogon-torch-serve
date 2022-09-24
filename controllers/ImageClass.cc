#include "controllers/ImageClass.h"

//add definition of your processing function here
Task<> ImageClass::classify(HttpRequestPtr req,
                            std::function<void(const HttpResponsePtr &)> callback) {
    Timer measure("ImageClass classify");

    auto start = std::chrono::high_resolution_clock::now();
    MultiPartParser fileUpload;
    Json::Value json;
    if (fileUpload.parse(req) != 0 || fileUpload.getFiles().size() != 1) {
        json["status"] = "failed";
        json["message"] = "Upload valid file for inference.";
        auto resp = HttpResponse::newHttpJsonResponse(json);
        resp->setStatusCode(k403Forbidden);
        callback(resp);
        co_return;
    }

    auto &file = fileUpload.getFiles()[0];
    auto data_view = file.fileContent();
    auto image_view = cv::Mat(1, data_view.length(), CV_8UC1, (void*) data_view.data());
    auto image = cv::imdecode(image_view, cv::IMREAD_COLOR);

    auto time_in_us = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - start);
    LOG_DEBUG << "File reading and cast to CV::Matrix executed in " << time_in_us.count() << "us";

    int randomIndex = rand() % batch_inference_engines.size();

    auto infer_start = std::chrono::high_resolution_clock::now();
    auto response = co_await batch_inference_engines[randomIndex]->infer(image);
    auto time_in_us_i = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::high_resolution_clock::now() - infer_start);
    LOG_DEBUG<< "Inference executed in " << time_in_us_i.count() << "us";

    json["status"] = "success";
    Json::Value prediction;
    prediction["label"] = response.className;
    prediction["confidence"] = response.confidence;
    json["prediction"] = prediction;
    auto resp = HttpResponse::newHttpJsonResponse(json);
    callback(resp);
    co_return;
}

ImageClass::ImageClass() {
    srand(time(nullptr));

    LOG_INFO << "Starting " << Configs::NUM_INFERENCE_THREADS << " inference engines.";

    for (int i = 0; i < Configs::NUM_INFERENCE_THREADS; ++i) {
        batch_inference_engines.emplace_back(std::make_unique<ModelBatchInference>());
        std::thread forever_infer([this, i]() {
            batch_inference_engines[i]->foreverBatchInfer();
        });
        forever_infer.detach();
    }
    // Load ModelBatchInference instead now
}
