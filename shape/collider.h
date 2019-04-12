#pragma once

#include "../maths/ray.h"

namespace Shape {
	struct Interface {
		float t;
		Math::Vector normal;
		Math::Point point;
	};

	class Collider {
	public:
		__device__
		virtual bool Hit(const Math::Ray& ray, Interface& interface, float tMin, float tMax) const = 0;
	};
}