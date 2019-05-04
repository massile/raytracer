#pragma once

#include "../image/image.h"

namespace Material {
	class Texture {
	public:
		__device__
		virtual Math::Vector UVProjection(float u, float v, const Math::Vector& point) const = 0;
	};
}