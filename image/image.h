#pragma once

#include <fstream>
#include "../maths/ray.h"

namespace Image {
	typedef Math::Vector Color;

	class Image {
	public:
		int width;
		int height;
		Color* pixels;

	public:
		Image(int width, int height) : width(width), height(height) {
			cudaMallocManaged(&pixels, width * height * sizeof(Color));
		}

		~Image() {
			cudaFree(pixels);
		}

		virtual void Write(const char* fileName) const = 0;
	};
}