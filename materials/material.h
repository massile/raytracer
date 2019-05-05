#pragma once

#include <curand_kernel.h>
#include "../maths/ray.h"
#include "../maths/operations.h"
#include "../image/ppm.h"
#include "./constant_texture.h"
#include "../shape/sphere.h"

namespace Material {
	class Material {
	public:
		__device__
		virtual bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const = 0;	
		
		__device__
		virtual Image::Color Emitted(const Math::Vector& point) const {
			return Image::Color(0.f, 0.f, 0.f);
		}
		
	};
}