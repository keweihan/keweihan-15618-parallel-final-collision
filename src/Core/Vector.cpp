#include "Core/Vector.h"
#include "math.h"
#include <cmath>

using namespace SimpleECS;
double SimpleECS::Vector::dotProduct(const Vector& other) const
{
    return other.x * this->x + other.y * this->y;
}

Vector SimpleECS::Vector::orthogonalVec() const
{
    return Vector(this->y, -this->x);
}

void SimpleECS::Vector::normalize()
{
    double magnitude = getMagnitude();
    this->x = this->x / magnitude;
    this->y = this->y / magnitude;
}

double SimpleECS::Vector::getMagnitude() const
{
    return sqrt(this->x * this->x + this->y * this->y);
}

double SimpleECS::Vector::distance(const Vector& other) const
{
    return std::sqrt( std::pow(other.x - this->x, 2) + std::pow(other.y - this->y, 2));
}

Vector SimpleECS::Vector::operator+(const Vector& other) const
{
    return Vector(this->x + other.x, this->y + other.y);
}

Vector SimpleECS::Vector::operator-(const Vector& other) const
{
    return Vector(this->x - other.x, this->y - other.y);
}

Vector SimpleECS::Vector::operator*(const double& other) const
{
    return Vector(this->x * other, this->y * other);
}

Vector SimpleECS::Vector::operator*(const int& other) const
{
    return Vector(this->x * other, this->y * other);
}