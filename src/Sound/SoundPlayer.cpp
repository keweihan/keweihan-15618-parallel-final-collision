#include "Sound/SoundPlayer.h"
#include "SDL_mixer.h"
#include <memory>

using namespace SimpleECS;

class SoundPlayer::SoundPlayerImpl
{
public:
	SoundPlayerImpl();
	~SoundPlayerImpl();

	Mix_Chunk* loadedAudio;
};

SoundPlayer::SoundPlayerImpl::SoundPlayerImpl()
{
	if (Mix_OpenAudio(44100, MIX_DEFAULT_FORMAT, 2, 2048) < 0)
	{
		printf("SDL_mixer could not initialize! SDL_mixer Error: %s\n", Mix_GetError());
	}
	loadedAudio = nullptr;
}

SoundPlayer::SoundPlayerImpl::~SoundPlayerImpl()
{
	Mix_FreeChunk(loadedAudio);

	// TODO: probably not intended to be called multiple times. 
	// May need to move to game loop.
	Mix_CloseAudio();
}

SoundPlayer::SoundPlayer(std::string pathToAudio)
{
	pImpl = std::make_unique<SoundPlayerImpl>();
	pImpl->loadedAudio = Mix_LoadWAV(pathToAudio.c_str());
}

SimpleECS::SoundPlayer::~SoundPlayer()
{
}

void SoundPlayer::playAudio()
{
	Mix_PlayChannel(-1, pImpl->loadedAudio, 0);
}

void SIMPLEECS_API SimpleECS::SoundPlayer::initialize()
{
	return void SIMPLEECS_API();
}

void SIMPLEECS_API SimpleECS::SoundPlayer::update()
{
	return void SIMPLEECS_API();
}

