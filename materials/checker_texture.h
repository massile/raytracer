#pragma once

#include "texture.h"

namespace Material {
	class CheckerTexture : public Texture {
	private:
		Texture* even;
		Texture* odd;
	public:
		__device__
		CheckerTexture(Texture* even, Texture* odd) : even(even), odd(odd) {}

		__device__
		Math::Vector UVProjection(float u, float v, const Math::Vector& point) const {
			float sine = sin(10 * point.x) * sin(10 * point.y) * sin(10 * point.z);
			if (sine < 0) {
				return odd->UVProjection(u, v, point);
			}
			return even->UVProjection(u, v, point);
		}

	};
}