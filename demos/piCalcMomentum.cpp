/*
Momentum based collision demo scene. Pi calculator. 

Based off concepts presented in the following video/paper, where it is proven
that in a system of a stationary object of mass 1, and object 100^n times the mass 
that collides with a the stationary object sending it to the wall, the amount of collisions
will be that of pi/10^n accurate to n digits.

https://www.youtube.com/watch?v=HEfHFsfGXjs&t=119s
https://www.maths.tcd.ie/~lebed/Galperin.%20Playing%20pool%20with%20pi.pdf

Kewei Han
*/
#include <SimpleECS.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <string>
#include <memory>
#include <cmath>

using namespace std;
using namespace SimpleECS;

// Ball parameters (try precision 3, SPEED = 5)
const int PRECISION = 0;
const int SPEED = 5;

// Environment parameters
const int SCREEN_HEIGHT = 720;
const int SCREEN_WIDTH = 1280;
const int WALL_THICKNESS = 50;
const int WEIGHT = pow(100, PRECISION);

// Globals
Scene* mainScene;

class CurrFrameCounter : public Component {
public:
	void initialize() {
		textRender = entity->getComponent<FontRenderer>();
	};

	void update() {
		frameCount++;
		int currSecond = static_cast<int>(Timer::getProgramLifetime() / 1000);
		if (currSecond > prevSecond)
		{
			displayFrames = frameCount;
			frameCount = 0;
			prevSecond = currSecond;
		};

		string text = "Current FPS: " + std::to_string(displayFrames);
		textRender->text = text;
	}
	uint64_t displayFrames = 0;
	uint64_t prevSecond = 0;
	uint64_t frameCount = 0;
	Handle<FontRenderer> textRender;
};

class CollisionCounterBall : public Component {
public:
	void initialize() {};
	
	void onCollide(const Collider& other) {
		collideCount++;
	}

	void update() { }
	uint64_t collideCount = 0;
	Handle<FontRenderer> textRender;
};

class CollisionCounterText : public Component {
public:
	CollisionCounterText(Handle<CollisionCounterBall> _ballCounter) {
		ballCounter = _ballCounter;
	}

	void initialize() {
		textRender = entity->getComponent<FontRenderer>();
	};

	void update() {
		string text = "Collision Count: " + std::to_string(ballCounter->collideCount);
		textRender->text = text;
	}
	Handle<CollisionCounterBall> ballCounter;
	Handle<FontRenderer> textRender;
};

Entity* createCurrFramesCounter()
{
	Entity* counter = mainScene->createEntity();
	counter->addComponent<FontRenderer>("Default", "assets/bit9x9.ttf", 26, Color(124, 200, 211, 0xff));
	counter->addComponent<CurrFrameCounter>();
	return counter;
}	
	
// Create ball with initial position and given velocity
Entity* createSquare(const int& x, const int& y, Vector vel, int side_length, double mass)
{
	Entity* newBall = mainScene->createEntity("ball");
	newBall->addComponent<RectangleRenderer>(side_length, side_length, Color(102, 102, 102, 102));
	newBall->addComponent<BoxCollider>(side_length, side_length);

	// Set position
	newBall->transform->position.x = x;
	newBall->transform->position.y = y;

	newBall->phys->velocity.x = vel.x;
	newBall->phys->velocity.y = vel.y;
	newBall->phys->mass = mass;

	return newBall;
}

// Create upper and lower walls in scene, and side walls that keep score tracking
void addBounds()
{
	Entity* topBound = mainScene->createEntity();
	topBound->addComponent<BoxCollider>(SCREEN_WIDTH + WALL_THICKNESS, WALL_THICKNESS);
	topBound->transform->position.y = SCREEN_HEIGHT / 2 + WALL_THICKNESS / 2;
	topBound->phys->is_static = true;
	topBound->phys->mass = 10000000;
	topBound->tag = "top";

	Entity* bottomBound = mainScene->createEntity();
	bottomBound->addComponent<BoxCollider>(SCREEN_WIDTH + WALL_THICKNESS, WALL_THICKNESS);
	bottomBound->transform->position.y = -SCREEN_HEIGHT / 2 - WALL_THICKNESS / 2;
	bottomBound->phys->is_static = true;
	bottomBound->phys->mass = 10000000;
	bottomBound->tag = "bottom";

	Entity* leftBound = mainScene->createEntity();
	leftBound->addComponent<BoxCollider>(WALL_THICKNESS, SCREEN_HEIGHT + WALL_THICKNESS);
	leftBound->transform->position.x = -SCREEN_WIDTH / 2 - WALL_THICKNESS / 2;
	leftBound->phys->is_static = true;
	leftBound->phys->mass = 10000000;
	leftBound->tag = "left";
}

int main() {
	try
	{
		cout << "Hello World!" << endl;

		// Create scene
		mainScene = new Scene(Color(0, 0, 0, 255));

		// Populate scene
		addBounds();

		// Create game with scene
		RenderConfig config;
		config.width = SCREEN_WIDTH;
		config.height = SCREEN_HEIGHT;
		config.gameName = "Pi Momentum Calculator";

		Game::getInstance().configure(config);
		Game::getInstance().addScene(mainScene);

		// Entity* floor = mainScene->createEntity();
		// floor->addComponent<LineRenderer>(Vector(- SCREEN_WIDTH / 2, -150), Vector(SCREEN_WIDTH / 2, -150), 3, Color(255, 255, 255, 1));

		// Create squares
		Entity* right = createSquare(-350, -75, { -SPEED, 0 }, 150, WEIGHT);
		right->addComponent<FontRenderer>("10e" + to_string(1 + PRECISION) + "kg", "assets/bit9x9.ttf", 26, Color(124, 200, 211, 0xff));

		Entity* left = createSquare(-500, -100, { 0, 0 }, 100, 1);
		Handle<CollisionCounterBall> ballCounter = left->addComponent<CollisionCounterBall>();
		left->addComponent<FontRenderer>("1kg", "assets/bit9x9.ttf", 26, Color(124, 200, 211, 0xff));
		left->getComponent<RectangleRenderer>()->renderColor = { 0, 33, 156, 1};
		
		// Create Collision Counter
		Entity* collideCounter = mainScene->createEntity();
		collideCounter->addComponent<FontRenderer>("Default", "assets/bit9x9.ttf", 26, Color(124, 200, 211, 0xff));
		collideCounter->addComponent<CollisionCounterText>(ballCounter);
		collideCounter->transform->position = { 0, 125 };

		Entity* framesCounter = createCurrFramesCounter();
		framesCounter->transform->position = { 0, 75 };

		// Start game loop
		Game::getInstance().startGame();
	}
	catch (const std::exception& e)
	{
		std::cout << "Caught exception: " << e.what() << std::endl;
	}
}