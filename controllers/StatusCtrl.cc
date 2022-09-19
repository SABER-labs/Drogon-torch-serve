#include "StatusCtrl.h"

void StatusCtrl::status(const HttpRequestPtr &req,
                        std::function<void(const HttpResponsePtr &)> &&callback) {
    LOG_DEBUG << "status check";
    auto resp = HttpResponse::newHttpResponse();
    resp->setBody("OK");
    callback(resp);
}