//
// Created by vigi99 on 16/01/22.
//

#pragma once
#include <chrono>
#include <utility>
#include <drogon/drogon.h>

class Timer {
public:
    explicit Timer(std::string);
    ~Timer();
private:
    std::chrono::system_clock::time_point start;
    std::string name_;
};
