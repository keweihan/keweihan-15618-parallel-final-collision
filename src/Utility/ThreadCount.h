#pragma once
#include "Core/Vector.h"
#include "Core/Timer.h"
#include <cstdint>

namespace UtilSimpleECS {
	class ThreadCount {
	public:
		static void logThreads(int val) {
            threadSum += val;
            frameCount++;
            timePassed += SimpleECS::Timer::getDeltaTime();
        }

        static uint32_t getAvgThreads() {
            timePassed = 0;
            return threadSum/frameCount;
        }

        static uint32_t timeSinceGetAvg() {
            return timePassed;
        }

    private:
        static uint32_t threadSum;
        static uint32_t frameCount;
        static uint32_t timePassed;
	};
}

uint32_t UtilSimpleECS::ThreadCount::threadSum = 0;
uint32_t UtilSimpleECS::ThreadCount::frameCount = 0;
uint32_t UtilSimpleECS::ThreadCount::timePassed = 0;