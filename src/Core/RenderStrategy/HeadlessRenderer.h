
#pragma once

#include "IRenderStrategy.h"

namespace UtilSimpleECS {
    class HeadlessRenderer : public IRenderStrategy {
    public:
        ~HeadlessRenderer() override;
        void init(const SimpleECS::RenderConfig& config) override;
        void frameBegin(SimpleECS::Scene* scene) override;
        void frameEnd(SimpleECS::Scene* scene) override;

        /// @brief Frequency in milliseconds to generate information
        const int INFO_FREQUENCY = 5000;

    private: 
        int prevOutputTime = 0;
        int framesSinceOutput = 0;
    };
}