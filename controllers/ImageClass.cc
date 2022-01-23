#include "controllers/ImageClass.h"
//add definition of your processing function here
Task<> ImageClass::classify(HttpRequestPtr req,
                            std::function<void(const HttpResponsePtr &)> callback)
{
    MultiPartParser fileUpload;
    Json::Value json;
    if (fileUpload.parse(req) != 0 || fileUpload.getFiles().size() != 1)
    {
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
    cv::Mat image_transformed;
    cv::resize(image, image_transformed, cv::Size(224, 224));
    cv::cvtColor(image_transformed, image_transformed, cv::COLOR_BGR2RGB);
    torch::Tensor tensor_image = torch::from_blob(image_transformed.data, {image_transformed.rows, image_transformed.cols, 3}, torch::kByte)
            .unsqueeze(0);
    int randomIndex = rand() % batch_inference_engines.size();
    auto response = co_await batch_inference_engines[randomIndex]->infer(tensor_image);
    json["status"] = "success";
    json["message"] = fmt::format("Class found for image was {} with confidence {:.{}f}.", response.className, response.confidence, 3);
    auto resp = HttpResponse::newHttpJsonResponse(json);
    callback(resp);
    co_return;
}

ImageClass::ImageClass()
{
    srand(time(nullptr));
    for (int i = 0; i < NUM_INFERENCE_ENGINES; ++i) {
        batch_inference_engines.emplace_back(std::make_unique<ModelBatchInference>());
        std::thread forever_infer([this, i]() {
            batch_inference_engines[i]->foreverBatchInfer();
        });
        forever_infer.detach();
    }
    // Load ModelBatchInference instead now
}
