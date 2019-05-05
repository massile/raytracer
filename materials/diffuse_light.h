#pragma once

#include "texture.h"

namespace Material {
	class DiffuseLight : public Material {
	private:
		Texture* emit;
	public:
		__device__
		DiffuseLight(Texture* emit) : emit(emit) {}

		__device__
		bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const {
			return false;
		}

		__device__
		Image::Color Emitted(const Math::Vector& point) const {
			return emit->UVProjection(0, 0, point);
		}
	};
}