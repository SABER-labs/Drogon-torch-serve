#include "controllers/ImageClass.h"

//add definition of your processing function here
Task<> ImageClass::classify(HttpRequestPtr req,
                            std::function<void(const HttpResponsePtr &)> callback) {
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

    int randomIndex = rand() % batch_inference_engines.size();
    auto response = co_await batch_inference_engines[randomIndex]->infer(image);

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

    auto num_inference_engines = getNumInferenceEngineThreads();
    LOG_INFO << "Starting " << num_inference_engines << " inference engines.";

    for (uint i = 0; i < num_inference_engines; ++i) {
        batch_inference_engines.emplace_back(std::make_unique<ModelBatchInference>());
        std::thread forever_infer([this, i]() {
            batch_inference_engines[i]->foreverBatchInfer();
        });
        forever_infer.detach();
    }
    // Load ModelBatchInference instead now
}
