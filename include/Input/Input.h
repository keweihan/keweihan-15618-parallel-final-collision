#pragma once

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	/**
	Defined keycodes for input
	*/
	enum class KeyCode
	{
		KEY_UP_ARROW = 82,
		KEY_DOWN_ARROW = 81,
		KEY_LEFT_ARROW = 80,
		KEY_RIGHT_ARROW = 79,

		KEY_W = 26,
		KEY_A = 4,
		KEY_S = 22,
		KEY_D = 7,
	};

	/**
	* Class containing methods for handling player input
	*/
	class SIMPLEECS_API Input {
	public:
		/**
		* Returns the current status of a key
		*
		* @param key : Keycode representing key status to query
		* @returns : false if released, true if pressed.
		*/
		static bool getKeyDown(KeyCode key);
	};
}