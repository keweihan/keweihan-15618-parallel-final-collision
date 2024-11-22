#include "./HeadlessRenderer.h"
#include "Core/Timer.h"
#include <iomanip> // Add at top

using namespace SimpleECS;
UtilSimpleECS::HeadlessRenderer::~HeadlessRenderer()
{
}

void UtilSimpleECS::HeadlessRenderer::init(const SimpleECS::RenderConfig &config)
{
    std::cout << "===== INFO =====" << std::endl;
    std::cout << "Running simulation in headless (no render) mode." << std::endl;
    std::cout << "Statistics logging every " << INFO_FREQUENCY << "ms" << std::endl;
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
        std::cout << std::fixed << std::setprecision(3)  // Set precision to 3 decimal places
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