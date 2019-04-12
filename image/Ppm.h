#pragma once

#include "image.h"

namespace Image {
	class Ppm : public Image {
	public:
		Ppm(int width, int height) : Image(width, height) {}

		void Write(const char* fileName) const override {
			std::ofstream file(fileName);
			file << "P3" << std::endl << width << ' ' << height << std::endl << 255 << std::endl;
			for (int j = height - 1; j >= 0; --j) {
				for (int i = 0; i < width; ++i) {
					Color pixel = pixels[i + j * width];
					file << int(pixel.r * 255.99f) << ' '
						 << int(pixel.g * 255.99f) << ' '
						 << int(pixel.b * 255.99f) << std::endl;
				}
			}
			file.close();
		}
	};
}