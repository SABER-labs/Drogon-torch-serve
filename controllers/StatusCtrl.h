#pragma once

#include <drogon/HttpController.h>

using namespace drogon;

class StatusCtrl : public drogon::HttpController<StatusCtrl> {
public:
    METHOD_LIST_BEGIN
        ADD_METHOD_TO(StatusCtrl::status, "/status", Get);
    METHOD_LIST_END

protected:
    static void status(const HttpRequestPtr &req, std::function<void(const HttpResponsePtr &)> &&callback);

};
