#include "TransformUtil.h"
#include "Core/GameRenderer.h"
#include "Core/Vector.h"
#include <cmath>

using namespace UtilSimpleECS;

SimpleECS::Vector TransformUtil::screenToWorldSpace(const int& x, const int& y)
{
	double worldX = x - (GameRenderer::SCREEN_WIDTH / 2);
	double worldY = y - GameRenderer::SCREEN_HEIGHT / 2;

	return SimpleECS::Vector(worldX, worldY);
}

SimpleECS::Vector TransformUtil::worldToScreenSpace(const double& x, const double& y)
{
	int screenX	= static_cast<int>(round(x + GameRenderer::SCREEN_WIDTH / 2));
	
	// -y due to inverted Y of SDL coordinates
	int screenY	= static_cast<int>(round(-y + GameRenderer::SCREEN_HEIGHT / 2));

	return SimpleECS::Vector(screenX, screenY);
}