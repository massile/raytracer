#pragma once

#include <curand_kernel.h>
#include "vector3d.h"

namespace Math {
	__device__
	Math::Vector RandomInSphere(curandState* rand) {
		Math::Vector direction;
		do {
			direction = 2.f * Math::Vector(curand_uniform(rand) - 0.5f, curand_uniform(rand) - 0.5f, curand_uniform(rand) - 0.5f);
		} while(Math::LengthSquared(direction) >= 1.0f);
		return direction;
	}

	__device__
	float SchlickPolynom(float cos, float iof) {
		float r0 = (1.f - iof) / (1.f + iof);
		r0 *= r0;
		return r0 + (1.f - r0) * pow(1.f - cos, 5);
	}
}