//
// Created by vigi99 on 16/01/22.
//
#pragma once

#include <chrono>
#include <utility>
#include <drogon/drogon.h>

class Timer {
public:
    explicit Timer(std::string name) : start(std::chrono::high_resolution_clock::now()), name_{name} {}

    ~Timer() {
        auto time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now() - start);
        LOG_INFO << name_ << " executed in " << time_in_ms.count() << "us";
    }

private:
    std::chrono::system_clock::time_point start;
    std::string name_;
};