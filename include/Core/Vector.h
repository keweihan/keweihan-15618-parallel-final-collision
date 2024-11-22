#pragma once

#include "SimpleECSAPI.h"

namespace SimpleECS
{
	/**
	* Representation of a 2D vector. Contains methods for common vector operations.
	*/
	class SIMPLEECS_API Vector
	{
	public:
		/**
		* Vector is by default 0,0.
		*/
		Vector() : x(0), y(0) {}
		Vector(double x, double y) : x(x), y(y) {}

		/**
		* Perform dot product
		* 
		* @param other Vector to perform dot operation with
		* @returns Returns dot product of this vector and 'other'
		*/
		double dotProduct(const Vector& other) const;

		/**
		* Retrieve a vector orthogonal (90 degree angle) with current vector.
		*
		* @returns Orthogonal vector. (x,y) will return (y, -x).
		*/
		Vector orthogonalVec() const;
		
		/**
		* Get magnitude of this vector.
		* @returns Magnitude of current vector.
		*/
		double getMagnitude() const;

		/**
		* Get distance between this vector as a point and other.
		* @returns distance between this vector and other.
		*/
		double distance(const Vector& other) const;

		/**
		* Change current vector to magnitude 1 without changing direction.
		*/
		void normalize();

		Vector operator+(const Vector& other) const;
		Vector operator-(const Vector& other) const;
		Vector operator*(const double& other) const;
		Vector operator*(const int& other) const;

		/**
		* x and y components member
		*/
		double x, y;
	};
}
