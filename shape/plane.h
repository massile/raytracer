#pragma once

#include "collider.h"

namespace Shape {
	class Plane : public Collider {
	private:
		Material::Material* material;
		Math::Point origin;
		float width;
		float height;
	public:
		__device__
		Plane(const Math::Point& origin, float width, float height, Material::Material* material) :
			origin(origin),
			width(width),
			height(height),
			material(material) {}

		__device__
		bool Hit(const Math::Ray& ray, Interface& interface, float tMin, float tMax) const {
			float t = (origin.z - ray.origin.z) / ray.direction.z;
			if (t < tMin || t > tMax) {
				return false;
			}

			Math::Point pt = ray(t);
			if (pt.x < origin.x || pt.x > origin.x + width || pt.y < origin.y || pt.y > origin.y + height) {
				return false;
			}

			interface.point = pt;
			interface.t = t;
			interface.material = material;
			interface.normal = Math::Vector(0, 0, 1);
			return true;
		}
	};
}