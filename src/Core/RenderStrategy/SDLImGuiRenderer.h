
#pragma once

#include "IRenderStrategy.h"

namespace UtilSimpleECS {
    class SDLImGuiRenderer : public IRenderStrategy {
    public:
        ~SDLImGuiRenderer() override;
        void init(const SimpleECS::RenderConfig& config) override;
        void frameBegin(SimpleECS::Scene* scene) override;
        void frameEnd(SimpleECS::Scene* scene) override;
    };
}