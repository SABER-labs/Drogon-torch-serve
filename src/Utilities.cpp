//
// Created by vigi99 on 16/01/22.
//

#include "Utilities.h"

#include <utility>

Timer::Timer(std::string name): name_(std::move(name)) {
    start = std::chrono::high_resolution_clock::now();
}

Timer::~Timer() {
    auto time_in_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start);
    LOG_INFO << name_ << " executed in " << time_in_ms.count() << "ms";
}