#pragma once

#include "SimpleECSAPI.h"

#include <cstdint>

namespace SimpleECS
{
	/**
	 * Representation of color with rgba values.
	 */
	class SIMPLEECS_API Color {

	public:
		// TODO: Hexadecimal constructor
		Color(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 0xff) : r(r), g(g), b(b), a(a) {};
		Color() : r(0), g(0), b(0), a(0xff) {};

		uint8_t r, g, b, a;
	};
}
