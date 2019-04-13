#pragma once

#include "../maths/ray.h"

namespace Material {
	class Material;
}

namespace Shape {
	struct Interface {
		float t;
		Math::Vector normal;
		Math::Point point;
		Material::Material *material;
	};

	class Collider {
	public:
		__device__
		virtual bool Hit(const Math::Ray& ray, Interface& interface, float tMin, float tMax) const = 0;
	};
}