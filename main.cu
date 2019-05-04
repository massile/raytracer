#include "renderer.h"

int main() {
	Image::Ppm image(1200, 600);
	Math::Point from(-3.2f, 0.8f, 3.2f);
	Math::Point to(0.f, 0.4f, 0.f);
	Math::Vector up(0.f, 1.0f, 0.0f);
	float focus = Math::Length(from - to) * 0.5f;

	Scene::Camera camera(.4f, from, to, up, 0.1f, focus);
	Renderer(&image, &camera).Run("test.ppm");
}	