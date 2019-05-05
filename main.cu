#include "renderer.h"

int main() {
	// Creates a PPM image
	Image::Ppm image(1200, 600);
	// Where the camera stands
	Math::Point from(-3.2f, 1.2f, 2.2f);
	// What the camera is looking at
	Math::Point to(0.2f, 0.4f, 0.f);
	// The up vector
	Math::Vector up(0.f, 1.0f, 0.0f);
	// Focal distance of the camera
	float focus = Math::Length(from - to);
	float aperture = 0.13f;

	Scene::Camera camera(.4f, from, to, up, aperture, focus);

	// Renders the image
	Renderer(&image, &camera).Run("test.ppm");
}	