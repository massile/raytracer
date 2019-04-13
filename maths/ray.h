#pragma once

#include "vector3d.h"

namespace Math {
	struct Ray {
		Point origin;
		Vector direction;

		Ray() = default;

		__device__
		Ray(const Point& origin, const Vector& direction) : origin(origin), direction(direction) {}

		__device__
		Point operator()(float t) const {
			return origin + t * direction;
		}
	};
}