#pragma once

#include "material.h"

namespace Material {
	class Dielectric : public Material {
	private:
		float iof;
	public:
		__device__
		Dielectric(float indexOfRefraction) : iof(indexOfRefraction) {}

		__device__
		bool Scatter(const Math::Ray& in, const Shape::Interface& surface, Image::Color& color, Math::Ray& out, curandState* random) const {
			Math::Vector normal;
			float refIndex;
			float cos;
			float inDotNormal = Math::Dot(in.direction, surface.normal);
			if (inDotNormal > 0) {
				normal = -surface.normal;
				refIndex = iof;
				cos = iof * inDotNormal / Math::Length(in.direction);
			} else {
				normal = surface.normal;
				refIndex = 1.0f / iof;
				cos = -inDotNormal / Math::Length(in.direction);
			}

			Math::Vector direction;
			bool shouldRefract = Math::Refract(in.direction, normal, refIndex, direction);
			float probaRefraction = shouldRefract ? Math::SchlickPolynom(cos, refIndex) : 1.0f;
			
			if (curand_uniform(random) < probaRefraction) {
				direction = Math::Reflect(in.direction, surface.normal);
			}

			out = Math::Ray(surface.point, direction);				
			color = Image::Color(1.f, 1.f, 1.f);
			return true;
		}
	};
}