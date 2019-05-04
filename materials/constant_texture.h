#pragma once

#include "./texture.h"

namespace Material {
	class ConstantTexture : public Texture {
	private:
		Image::Color color;
	public:
		__device__
		ConstantTexture(const Image::Color& color) : color(color) {}

		__device__
		Math::Vector UVProjection(float u, float v, const Math::Vector& point) const {
			return color;
		}
	};
}