#include "Physics/PhysicsBody.h"
#include "Core/Timer.h"
#include <iostream>

void SimpleECS::PhysicsBody::update()
{
	//Vector gravity = Vector(0, -90.86);
	//acceleration = gravity;

	velocity.x += acceleration.x * Timer::getDeltaTime() / 1000;
	velocity.y += acceleration.y * Timer::getDeltaTime() / 1000;

	entity->transform->position.x += velocity.x * Timer::getDeltaTime()/1000;
	entity->transform->position.y += velocity.y * Timer::getDeltaTime()/1000;

	futureVelocity = velocity;

}

void SimpleECS::PhysicsBody::applyForce(Vector direction, double magnitude)
{
	direction.normalize();
	velocity.x += (direction * magnitude * (1 / mass)).x;
	velocity.y += (direction * magnitude * (1 / mass)).y;
}

void SimpleECS::PhysicsBody::onCollide(const Collider& other)
{

}

void SimpleECS::PhysicsBody::onCollide(const Collision& collide)
{
	// TODO:
	// Eventaully collision resolution needs to all occur at the SAME TIME AT END OF FRAME
	// i.e. setting first before getting. 
	// 
	// Consider two physics body going towards each other in
	// in a collision. If the first body's velocity is set from collision resolution
	// before the second resolves its collision, the second will resolve as if the first was already
	// flying away.	

	// Calculate new velocity (simple reflect across collision normal)
	// velocity = velocity - (collide.normal * (velocity.dotProduct(collide.normal))) * 2;
	// futureVelocity = velocity;

	PhysicsBody other;
	double massCoef;
	try {
		other = *collide.b->entity->getComponent<PhysicsBody>();
		massCoef = 2 * other.mass / (other.mass + mass);
	}
	catch (const std::exception&){
		other = PhysicsBody();
		other.velocity = Vector();
		massCoef = 2;
	}

	// Shift position out of overlap (weighted shift amount based on relative mass)
	entity->transform->position.x += (collide.normal * collide.penetration).x * massCoef/2;
	entity->transform->position.y += (collide.normal * collide.penetration).y * massCoef/2;

	// Calculate new velocity (mass velocity 2D calculation)
	// Adapted from https://en.wikipedia.org/wiki/Elastic_collision#Two-Dimensional_Collision_With_Two_Moving_Objects
	// using collision normals instead of position vectors 
	Vector posDiff		= entity->transform->position - collide.b->entity->transform->position;
	double posDiffMag	= posDiff.getMagnitude() * posDiff.getMagnitude();
	double dotProd		= posDiff.dotProduct(velocity - other.velocity);
	Vector velocityChange = collide.normal * ((velocity - other.velocity).dotProduct(collide.normal)) * massCoef;

	futureVelocity = velocity - velocityChange;
}

void SimpleECS::PhysicsBody::lateUpdate()
{
	velocity = futureVelocity;
}
