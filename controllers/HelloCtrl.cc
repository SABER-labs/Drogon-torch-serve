#include "HelloCtrl.h"
#include <fmt/core.h>
//add definition of your processing function here
void HelloCtrl::hello(const HttpRequestPtr &req,
                 std::function<void (const HttpResponsePtr &)> &&callback)
{
    LOG_DEBUG << "Hello world was called.";
    auto resp = HttpResponse::newHttpResponse();
    resp->setBody("Hello world!");
    callback(resp);
}
void HelloCtrl::hello_name(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback,
                         const std::string& name)
{
    LOG_DEBUG << name << " was passed.";
    auto resp = HttpResponse::newHttpResponse();
    resp->setBody(fmt::format("Hello {}!", name));
    callback(resp);
}