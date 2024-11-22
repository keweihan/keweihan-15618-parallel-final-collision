#include "Core/Timer.h"
#include <SDL.h>
#include <iostream>

uint64_t SimpleECS::Timer::frameFinishTime;
uint16_t SimpleECS::Timer::previousFrameLength;

bool SimpleECS::Timer::inFreezeMode = false;
bool SimpleECS::Timer::isFreezed = false;
uint16_t SimpleECS::Timer::freezeFrameLength;

uint64_t SimpleECS::Timer::getDeltaTime()
{
	if (inFreezeMode) {
		return isFreezed ? 0 : freezeFrameLength; 
	}

	// TODO: move to config file setup
	int maxDeltaTime = 500;
	return previousFrameLength < maxDeltaTime ? previousFrameLength : maxDeltaTime;
}

uint64_t SimpleECS::Timer::getProgramLifetime()
{
	return SDL_GetTicks64();
}

void SimpleECS::Timer::setFreezeMode(bool freeze)
{
	inFreezeMode = freeze;
	isFreezed = freeze;
}

void SimpleECS::Timer::freezeStep(uint16_t time)
{
	if (inFreezeMode)
	{
		isFreezed = false;
		freezeFrameLength = time;
	}
	else
	{
		// todo: throw warning
	}
}


void SimpleECS::Timer::endFrame()
{
	// Normal logic
	uint64_t currTicks = SDL_GetTicks64();
	previousFrameLength = static_cast<uint16_t>(currTicks - frameFinishTime);

	frameFinishTime = currTicks;

	// Freeze logic
	isFreezed = inFreezeMode;
}