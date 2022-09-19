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
    LOG_DEBUG << "Classify function was called.";
    std::vector<char> data(file.fileData(), file.fileData() + file.fileLength());
    auto image = cv::imdecode(cv::Mat(data), cv::ImreadModes::IMREAD_COLOR);

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

    char * val = getenv( "NUM_INFERENCE_ENGINES" );
    uint num_inference_engines = val == nullptr ? (std::thread::hardware_concurrency() / 5) : std::stoi(val);

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
