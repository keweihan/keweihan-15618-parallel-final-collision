#include "./HeadlessRenderer.h"
#include "Core/Timer.h"
#include "Collision/ColliderSystem.h"
#include <iomanip> // Add at top

using namespace SimpleECS;
UtilSimpleECS::HeadlessRenderer::~HeadlessRenderer()
{
}

std::string gridSchemeToString(GridScheme scheme) {
    switch (scheme) {
        case QUADTREE_CUDA: return "QUADTREE_CUDA";
        case QUADTREE_SEQ: return "QUADTREE_SEQ";
        case STATIC_GRID_CUDA: return "STATIC_GRID_CUDA";
        case STATIC_GRID_SEQ: return "STATIC_GRID_SEQ";
        case QUADTREE_CUDA_SEMI_STATIC: return "QUADTREE_CUDA_SEMI_STATIC";
        default: return "UNKNOWN_SCHEME";
    }
}

void UtilSimpleECS::HeadlessRenderer::init(const SimpleECS::RenderConfig &config)
{
    std::string gridScheme = gridSchemeToString(ColliderSystem::getInstance().scheme);
    std::cout << "===== INFO =====" << std::endl;
    std::cout << "Running simulation in headless (no render) mode." << std::endl;
    std::cout << "Statistics logging every " << INFO_FREQUENCY << "ms" << std::endl;
    std::cout << "Running with collision scheme " << gridScheme << std::endl;
    std::cout << "Change INFO_FREQUENCY variable in HeadlessRenderer.h to configure" << std::endl;
}

void UtilSimpleECS::HeadlessRenderer::frameBegin(SimpleECS::Scene* scene)
{

}

void UtilSimpleECS::HeadlessRenderer::frameEnd(SimpleECS::Scene* scene)
{
    int currentTime = SimpleECS::Timer::getProgramLifetime();
    if(currentTime - prevOutputTime >= INFO_FREQUENCY)
    {
        std::cout << std::fixed << std::setprecision(7)  // Set precision to 3 decimal places
                  << "====  Performance Stats @ "  << static_cast<int>(currentTime/1000) << "s  ====" << std::endl
                  << "Total entities         : " << scene->getEntities().size() << std::endl
                  << "Steps/frames simulated : " << framesSinceOutput << std::endl
                  << "Avg FPS                : " << static_cast<double>(framesSinceOutput) / (INFO_FREQUENCY / 1000.0) << std::endl
                  << "Avg ms per frame       : " << static_cast<double>(INFO_FREQUENCY) / framesSinceOutput << std::endl
                  << std::endl;

        prevOutputTime = SimpleECS::Timer::getProgramLifetime();
        framesSinceOutput = 0;
    }
    else
    {
        framesSinceOutput++;
    }
}