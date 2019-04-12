#pragma once

#include "image/ppm.h"
#include "scene/camera.h"
#include "shape/sphere.h"
#include "shape/container.h"

__device__
Image::Color ComputeColor(const Math::Ray& ray, Shape::Collider* collider);

__global__
void InitColliders(Shape::Collider** container, Shape::Collider** spheres, int size) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	spheres[0] = new Shape::Sphere(Math::Point(0, 0.0f, -1.f), 0.5f);
	*container = new Shape::Container(spheres, size);
}

__global__
void Render(Image::Color* pixels, int width, int height, Scene::Camera* camera, Shape::Collider** collider) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	float u = float(x) / width;
	float v = float(y) / height;

	pixels[x + y * width] = ComputeColor(camera->RayAt(u, v), *collider);
}