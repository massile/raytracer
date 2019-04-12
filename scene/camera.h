#pragma once

#include "../maths/ray.h"

namespace Scene {
	class Camera {
	private:
		Math::Vector right;
		Math::Vector up;
		Math::Point origin;
		Math::Point center;
	public:
		__host__ __device__
		Camera(float fovY, const Math::Point& eye, const Math::Point& at, const Math::Vector& vUp) : center(at) {
			float halfHeight = tan(fovY);
			float halfWidth = 2.f * halfHeight;

			Math::Vector z = Math::Normalize(at - eye);
			Math::Vector x = Math::Cross(z, vUp);
			Math::Vector y = Math::Cross(x, z);

			right = 2.f * halfWidth * x;
			up = 2.f * halfHeight * y;
			origin = -halfHeight * y - halfWidth * x - z;
		}

		__device__
		Math::Ray RayAt(float u, float v) {
			return Math::Ray(center, origin + u * right + v * up - center);
		}
	};
}