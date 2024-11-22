#pragma once
#ifndef TIME_H
#define TIME_H

#include <cstdint>

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	class Game;

	/**
	Provides time related information
	*/
	class SIMPLEECS_API Timer {
	public:
		// Game has access to endFrame()
		friend class Game;

		/*
		* Milliseconds since last frame execution finish 
		* (frame 'finish' time excludes time to render).
		*/
		static uint64_t getDeltaTime();

		/*
		* Milliseconds since program started
		*/
		static uint64_t getProgramLifetime();

		/*
		* Freeze time (getDeltaTime returns 0)
		* TODO: separate time and physics time.
		*/
		static void setFreezeMode(bool);

		/*
		 * Step forward one step in time.
		 * TODO: separate time and physics time.
		 */
		static void freezeStep(uint16_t time);

	private:
		// Time of previous frame finish in ms
		static uint64_t frameFinishTime;

		// Duration of previous frame
		static uint16_t previousFrameLength;

		// Time is in freeze mode
		static bool inFreezeMode; // Dictates isFrozen == true
		static bool isFreezed;	  // Dictates getDeltaTime return 0
		static uint16_t freezeFrameLength; // Length of frozen step

		/*
		* Call at end of every game loop to indicate end
		*/
		static void endFrame();
	};
}

#endif
