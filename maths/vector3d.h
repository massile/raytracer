#pragma once

namespace Math {
	struct Vector {
		union {
			struct {
				float r;
				float g;
				float b;
			};
			struct {
				float x;
				float y;
				float z;
			};
		};

		Vector() = default;

		__host__ __device__
		Vector(float x, float y, float z) : x(x), y(y), z(z) {}

		__host__ __device__
		Vector operator+(const Vector& v) const {
			return Vector(x + v.x, y + v.y, z + v.z);
		}

		__host__ __device__
		Vector operator+=(const Vector& v) {
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}

		__host__ __device__
		Vector operator-(const Vector& v) const {
			return Vector(x - v.x, y - v.y, z - v.z);
		}

		__host__ __device__
		Vector operator-=(const Vector& v) {
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}

		__host__ __device__
		Vector operator*(const Vector& v) const {
			return Vector(x * v.x, y * v.y, z * v.z);
		}

		__host__ __device__
		Vector operator*=(const Vector& v) {
			x *= v.x;
			y *= v.y;
			z *= v.z;
			return *this;
		}

		__host__ __device__
		Vector operator*(float a) const {
			return Vector(x * a, y * a, z * a);
		}

		__host__ __device__
		Vector operator*=(float a) {
			x *= a;
			y *= a;
			z *= a;
			return *this;
		}

		__host__ __device__
		Vector operator/(float a) const {
			float invA = 1.0f / a;
			return Vector(x * invA, y * invA, z * invA);
		}

		__host__ __device__
		Vector operator/=(float a) {
			float invA = 1.0f / a;
			x *= invA;
			y *= invA;
			z *= invA;
			return *this;
		}
	};

	typedef Vector Point;

	__host__ __device__
	inline Vector operator*(float a, const Vector& v) {
		return v * a;
	}

	__host__ __device__
	inline Vector Lerp(float t, const Vector& u, const Vector& v) {
		return (1.f - t) * u + t * v;
	}

	__host__ __device__
	inline float Dot(const Vector& u, const Vector& v) {
		return u.x * v.x + u.y * v.y + u.z * v.z;
	}

	__host__ __device__
	inline Vector Cross(const Vector& u, const Vector& v) {
		return Vector(
			u.y * v.z - u.z * v.y,
			u.z * v.x - u.x * v.z,
			u.x * v.y - u.y * v.x
		);
	}

	__host__ __device__
	inline float LengthSquared(const Vector& v) {
		return Dot(v, v);
	}

	__host__ __device__
	inline float Length(const Vector& v) {
		return sqrt(LengthSquared(v));
	}

	__host__ __device__
	inline Vector Normalize(const Vector& v) {
		return v / Length(v);
	}
}