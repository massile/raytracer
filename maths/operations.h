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
}