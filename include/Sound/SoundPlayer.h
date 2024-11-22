#pragma once
#include "Core/Component.h"
#include "Utility/Color.h"
#include <string>
#include <memory>


#include "SimpleECSAPI.h"

namespace SimpleECS
{
	/**
	Stores reference to audio file and allows playing audio. 
	*/
	class SoundPlayer : public Component
	{
	public:
		SIMPLEECS_API SoundPlayer(std::string pathToAudio);
		SIMPLEECS_API ~SoundPlayer();
		/**
		Plays associated audio clip once.
		*/
		SIMPLEECS_API void playAudio();
		std::string path = "";

		void SIMPLEECS_API initialize() override;
		void SIMPLEECS_API update() override;
	private:
		class SoundPlayerImpl;
		std::unique_ptr<SoundPlayerImpl> pImpl;
	};
}
