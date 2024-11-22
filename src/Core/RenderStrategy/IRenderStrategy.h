// IRenderStrategy.h
#pragma once
#include <Core/RenderConfig.h>
#include <Core/Scene.h>

namespace UtilSimpleECS {
    class IRenderStrategy {
    public:
        virtual ~IRenderStrategy() = default;
        virtual void init(const SimpleECS::RenderConfig& config) = 0;
        virtual void frameBegin(SimpleECS::Scene* scene) = 0;
        virtual void frameEnd(SimpleECS::Scene* scene) = 0;
    };
}