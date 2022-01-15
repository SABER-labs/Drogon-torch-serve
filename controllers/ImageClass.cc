#include "controllers/ImageClass.h"
//add definition of your processing function here
void ImageClass::classify(const HttpRequestPtr &req,
                      std::function<void (const HttpResponsePtr &)> &&callback)
{
    auto nograd = torch::NoGradGuard();
    LOG_INFO << "Classify function was called.";
    auto response_string = std::string("");
    if (torch::cuda::is_available()) {
        std::vector<torch::jit::IValue> inputs;
        auto image = cv::imread("../model_resources/cat.jpg", cv::ImreadModes::IMREAD_COLOR);
        cv::Mat image_transformed;
        cv::resize(image, image_transformed, cv::Size(224, 224));
        cv::cvtColor(image_transformed, image_transformed, cv::COLOR_BGR2RGB);
        torch::Tensor tensor_image = torch::from_blob(image_transformed.data, {image_transformed.rows, image_transformed.cols, 3}, torch::kByte);
        tensor_image = tensor_image.to(torch::kCUDA);
        tensor_image = tensor_image.permute({2, 0, 1});
        tensor_image = tensor_image.toType(torch::kFloat);
        tensor_image = tensor_image.div(255);
        tensor_image = tensor_image.unsqueeze(0);
        inputs.push_back(torch::autograd::make_variable(tensor_image, false));
        auto output = model.forward(inputs).toTensor();
        auto values = output.argmax(1);
        auto confidence = torch::softmax(output, 1).max().item<float>();
        response_string = fmt::format("Class found for image was {} with confidence {}.", class_idx_to_names[std::to_string(values.cpu().item<int>())], confidence);
    }
    auto resp = HttpResponse::newHttpResponse();
    resp->setBody(response_string);
    callback(resp);
}

ImageClass::ImageClass()
{
    LOG_INFO << "Model was created.";
    model = torch::jit::load(std::filesystem::absolute("../model_resources/resnet18_traced.pt"));
    std::ifstream i("../model_resources/class_names.json");
    i >> class_idx_to_names;
    torch::NoGradGuard();
    model.eval();
    if (torch::cuda::is_available()) {
        LOG_INFO << "Model loaded onto cuda.";
        model.to(torch::kCUDA);
    }
}
