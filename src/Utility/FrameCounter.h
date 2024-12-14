#pragma once
#include "Core/Vector.h"
#include "Core/Timer.h"
#include <cstdint>

namespace UtilSimpleECS {
	class FrameCounter {
	public:
        FrameCounter() {}

		void countFrame() {
            frameCount++;
            timePassed += SimpleECS::Timer::getDeltaTime();
        }

        void resetCounter() {
            frameCount = 0;
            timePassed = 0;
        }

        uint32_t getFrameCount() {
            return frameCount;
        }

        uint32_t getTime() {
            return timePassed;
        }

    private:
        uint32_t frameCount = 0;
        uint32_t timePassed = 0;
	};
}