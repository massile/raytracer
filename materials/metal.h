#pragma once

#include "material.h"

namespace Material {
	class Metal : public Material {
	private:
		Image::Color albedo;
		float fuzziness;
	public:
		__device__
		Metal(const Image::Color& albedo, float fuzziness = 0.0f) : albedo(albedo), fuzziness(fuzziness) {}

		__device__
		bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const {
			Math::Vector reflection = Math::Reflect(Math::Normalize(in.direction), surface.normal);
			out = Math::Ray(surface.point, reflection + fuzziness * Math::RandomInSphere(random));
			color = albedo;
			return Math::Dot(out.direction, surface.normal) > 0;
		}
	};
}