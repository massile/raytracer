#pragma once

#include <curand_kernel.h>
#include "../maths/ray.h"
#include "../maths/operations.h"
#include "../image/ppm.h"
#include "../shape/sphere.h"

namespace Material {
	class Material {
	public:
		__device__
		virtual bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const = 0;	
	};
}