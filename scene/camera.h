#pragma once

#include "../maths/operations.h"
#include "../maths/ray.h"

namespace Scene {
	class Camera {
	private:
		Math::Vector x;
		Math::Vector y;
		Math::Vector right;
		Math::Vector up;
		Math::Point origin;
		Math::Point center;
		float lensRadius;
	public:
		__host__ __device__
		Camera(float fovY, const Math::Point& eye, const Math::Point& at, const Math::Vector& vUp, float aperture, float focus)
			: center(eye), lensRadius(aperture * 0.5f) {
			float halfHeight = tan(fovY);
			float halfWidth = 2.f * halfHeight;

			Math::Vector z = Math::Normalize(eye - at);
			x = Math::Cross(vUp, z);
			y = Math::Cross(z, x);

			right = 2.f * focus * halfWidth * x;
			up = 2.f * focus * halfHeight * y;
			origin = eye - focus * (halfHeight * y + halfWidth * x + z);
		}

		__device__
		Math::Ray RayAt(float u, float v, curandState* randState) {
			Math::Vector rand = lensRadius * Math::RandomInSphere(randState);
			Math::Vector offset = x * rand.x + y * rand.y;
			return Math::Ray(center + offset, origin + u * right + v * up - center - offset);
		}
	};
}