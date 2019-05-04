#pragma once

#include "material.h"

namespace Material {
	class Lambert : public Material {
	private:
		Texture* albedo;

	public:
		__device__
		Lambert(Texture* albedo) : albedo(albedo) {}

		__device__
		bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const override {
			color = albedo->UVProjection(0, 0, surface.point);
			out = Math::Ray(surface.point, surface.normal + Math::RandomInSphere(random));
			return true;
		}
	};
}