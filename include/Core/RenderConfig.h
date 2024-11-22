#pragma once
#include <string>

namespace SimpleECS {
    struct RenderConfig {
        std::string gameName = "SimpleECS"; 
        bool enableWindow = true;
        int width = 640;
        int height = 480;
    };
}