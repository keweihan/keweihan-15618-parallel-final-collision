#include "Render/LineRenderer.h"
#include "Core/GameRenderer.h"
#include "Utility/TransformUtil.h"
#include <tuple>
#include <cmath>
#include "SDL.h"

using namespace SimpleECS;
using namespace UtilSimpleECS;

void LineRenderer::update()
{
	if(GameRenderer::renderer == nullptr) {
        static bool errorPrinted = false;
        if(!errorPrinted) {
            printf("Warning: GameRenderer::renderer is null. LineRenderer components will not render.\n");
            errorPrinted = true;
        }
        return;
    }

	if (spacing == 0)
	{
		drawECSLine(start, end);
	}
	else
	{
		Vector line(end.x - start.x, end.y - start.y);
		double slope = line.y / line.x;

		int xRise = static_cast<int>(sqrt(spacing * spacing / (slope * slope + 1)));
		int yRise = static_cast<int>(sqrt(spacing * spacing / (1/(slope * slope) + 1)));
		Vector currStart = start;
		Vector nextEnd = Vector(currStart.x + xRise, currStart.y + yRise);

		while (nextEnd.distance(end) > spacing * 2)
		{
			drawECSLine(currStart, nextEnd);
			currStart = Vector(nextEnd.x + xRise, nextEnd.y + yRise);
			nextEnd = Vector(currStart.x + xRise, currStart.y + yRise);
		}
	}
}

void SimpleECS::LineRenderer::drawECSLine(Vector startPoint, Vector endPoint)
{
	Vector line(endPoint.x - startPoint.x, endPoint.y - startPoint.y);
	Vector orth = line.orthogonalVec();
	orth.normalize();

	for (int i = 0; i < width; i++)
	{
		int xOffset = static_cast<int>(-orth.x * width / 2 + i * (orth.x));
		int yOffset = static_cast<int>(-orth.y * width / 2 + i * (orth.y));

		auto startCoord = TransformUtil::worldToScreenSpace(startPoint.x + xOffset, startPoint.y + yOffset);
		auto endCoord = TransformUtil::worldToScreenSpace(endPoint.x + xOffset, endPoint.y + yOffset);
		SDL_SetRenderDrawColor(GameRenderer::renderer, renderColor.r, renderColor.g, renderColor.b, renderColor.a);
		SDL_RenderDrawLine(GameRenderer::renderer, static_cast<int>(startCoord.x), static_cast<int>(startCoord.y), 
						   static_cast<int>(endCoord.x), static_cast<int>(endCoord.y));
	}
}
