#pragma once

#include "collider.h"

namespace Shape {
	class Sphere : public Collider {
	public:
		Math::Point center;
		Material::Material* material;
		float radius;

	public:
		__device__
		Sphere(const Math::Point& center, float radius, Material::Material* material) :
			center(center), radius(radius), material(material) {}

		__device__
		bool Hit(const Math::Ray& ray, Interface& interface, float tMin, float tMax) const override {
			Math::Vector oc = ray.origin - center;
			float a = Math::LengthSquared(ray.direction);
			float b = 2.f * Math::Dot(oc, ray.direction);
			float c = Math::LengthSquared(oc) - radius * radius;

			float discrim = b*b - 4.f * a * c;
			if (discrim < 0.f) {
				return false;
			}

			float sqrtD = sqrt(discrim);
			float t = (-b - sqrtD)/(2.f * a);
			if (t < tMin || t > tMax) {
				t += sqrtD / a;
			}
			if (t < tMin || t > tMax) {
				return false;
			}
			
			interface.t = t;
			interface.point = ray(t);
			interface.normal = (interface.point - center) / radius;
			interface.material = material;
			
			return true;
		}
	};
}