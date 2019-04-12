#include "renderer.h"

int main() {
	Image::Ppm image(1200, 600);
	Scene::Camera camera(.7f, Math::Point(0.f, 0.f, -1.0f), Math::Point(0.f, 0.f, 0.f), Math::Vector(0.f, 1.0f, 0.0f));
	Renderer(&image, &camera).Run("test.ppm");
}