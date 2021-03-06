//
// Created by vigi99 on 16/01/22.
//
#pragma once
#include <chrono>
#include <utility>
#include <drogon/drogon.h>

class Timer {
public:
    explicit Timer(std::string): start(std::chrono::high_resolution_clock::now()) {}
    ~Timer() {
        auto time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
        LOG_INFO << name_ << " executed in " << time_in_ms.count() << "ms";
    }
private:
    std::chrono::system_clock::time_point start;
    std::string name_;
};