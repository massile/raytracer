#pragma once

#include "material.h"

namespace Material {
	class Lambert : public Material {
	private:
		Image::Color albedo;

	public:
		__device__
		Lambert(const Image::Color& albedo) : albedo(albedo) {}

		__device__
		bool Scatter(const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const override {
			color = albedo;
			out = Math::Ray(surface.point, surface.normal + Math::RandomInSphere(random));
			return true;
		}
	};
}