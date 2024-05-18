// timer.h

#ifndef TIMER_H
#define TIMER_H

#include <chrono>

class Timer {
public:
    Timer() : startTime(), endTime() {}

    // Start the timer
    void start() {
        startTime = std::chrono::high_resolution_clock::now();
    }

    // Stop the timer
    void stop() {
        endTime = std::chrono::high_resolution_clock::now();
    }

    // Get the elapsed time in milliseconds
    double elapsed() const {
        std::chrono::duration<double, std::milli> duration = endTime - startTime;
        return duration.count();
    }

private:
    // Time points to store the start and end times
    std::chrono::high_resolution_clock::time_point startTime;
    std::chrono::high_resolution_clock::time_point endTime;
};

#endif // TIMER_H
