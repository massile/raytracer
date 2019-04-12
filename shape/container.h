#pragma once

#include "collider.h"

namespace Shape {
	class Container : public Collider {
	private:
		Collider** shapes;
		int size;
	public:
		__device__
		Container(Collider** shapes, int size) : shapes(shapes), size(size) {}

		__device__
		bool Hit(const Math::Ray& ray, Interface& interface, float tMin, float tMax) const override {
			Interface tmpInterface;
			bool hasHit = false;
			float closest = tMax;
			for (int i = 0; i < size; ++i) {
				if (shapes[i]->Hit(ray, tmpInterface, tMin, closest)) {
					hasHit = true;
					closest = tmpInterface.t;
					interface = tmpInterface;
				}
			}
			return hasHit;
		}
	};
}