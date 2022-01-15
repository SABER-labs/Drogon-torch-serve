#pragma once
#include <drogon/HttpController.h>
using namespace drogon;
class HelloCtrl:public drogon::HttpController<HelloCtrl>
{
  public:
    METHOD_LIST_BEGIN
    //use METHOD_ADD to add your custom processing function here;
    //METHOD_ADD(HelloCtrl::get,"/{2}/{1}",Get);//path is /HelloCtrl/{arg2}/{arg1}
    //METHOD_ADD(HelloCtrl::your_method_name,"/{1}/{2}/list",Get);//path is /HelloCtrl/{arg1}/{arg2}/list
    ADD_METHOD_TO(HelloCtrl::hello, "/hello", Get);
    ADD_METHOD_TO(HelloCtrl::hello_name, "/hello_name?name={1}", Get);
    //ADD_METHOD_TO(HelloCtrl::your_method_name,"/absolute/path/{1}/{2}/list",Get);//path is /absolute/path/{arg1}/{arg2}/list

    METHOD_LIST_END
    // your declaration of processing function maybe like this:
    // void get(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback,int p1,std::string p2);
    // void your_method_name(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback,double p1,int p2) const;
 protected:
    static void hello(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback);
    static void hello_name(const HttpRequestPtr& req,std::function<void (const HttpResponsePtr &)> &&callback, const std::string& name);

};
