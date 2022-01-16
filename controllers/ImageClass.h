#pragma once
#include <fmt/core.h>
#include <drogon/HttpController.h>
#include <c10/cuda/CUDAGuard.h>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include "includes/json.hpp"
#include "src/ModelBatchInference.h"
using namespace drogon;
class ImageClass:public drogon::HttpController<ImageClass>
{
  public:
    METHOD_LIST_BEGIN
    //use METHOD_ADD to add your custom processing function here;
    //METHOD_ADD(ImageClass::get,"/{2}/{1}",Get);//path is /ImageClass/{arg2}/{arg1}
    //METHOD_ADD(ImageClass::your_method_name,"/{1}/{2}/list",Get);//path is /ImageClass/{arg1}/{arg2}/list
    //ADD_METHOD_TO(ImageClass::your_method_name,"/absolute/path/{1}/{2}/list",Get);//path is /absolute/path/{arg1}/{arg2}/list
    ADD_METHOD_TO(ImageClass::classify, "/classify", Post);

    METHOD_LIST_END
    // your declaration of processing function maybe like this:
    // void get(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback,int p1,std::string p2);
    // void your_method_name(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback,double p1,int p2) const;
    ImageClass();
    const static int NUM_INFERENCE_ENGINES = 3;
private:
    std::vector<std::unique_ptr<ModelBatchInference>> batch_inference_engines;
protected:
    void classify(const HttpRequestPtr& req, std::function<void (const HttpResponsePtr &)> &&callback);
};
