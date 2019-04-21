#pragma once

#include "scene/camera.h"
#include "materials/lambert.h"
#include "materials/dielectric.h"
#include "materials/metal.h"
#include "shape/container.h"

__device__
Image::Color ComputeColor(const Math::Ray& ray, Shape::Collider* collider, curandState* random, int depth = 0);

__global__
void InitColliders(Shape::Collider** container, Shape::Collider** spheres, Material::Material** materials, int size) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;
	materials[0] = new Material::Metal(Image::Color(0.4f, 0.4f, 0.3f), 1.f);
	materials[1] = new Material::Metal(Image::Color(1.0f, 0.4f, 0.3f), .9f);
	materials[2] = new Material::Metal(Image::Color(0.3f, 0.2f, 1.0f), .8f);
	materials[3] = new Material::Metal(Image::Color(0.4f, 0.3f, 1.0f), .2f);
	materials[4] = new Material::Metal(Image::Color(1.0f, 1.0f, 1.0f), .1f);
	materials[5] = new Material::Lambert(Image::Color(0.8f, 0.7f, 0.3f));
	materials[6] = new Material::Lambert(Image::Color(0.3f, 0.8f, 1.0f));
	materials[7] = new Material::Dielectric(1.49f);
	materials[8] = new Material::Dielectric(1.51f);
	materials[9] = new Material::Lambert(Image::Color(0.8f, 0.8f, 0.8f));

	for (int j = 0; j < 3; ++j)	{
		for (int i = 0; i < 3; ++i) {
			int index = i+3*j;
			spheres[index] = new Shape::Sphere(Math::Point((i-1) * 0.8f, 0, (j-1) * 0.8f), 0.3f, materials[index]);			
		}
	}
	spheres[9] = new Shape::Sphere(Math::Point(0, -100.3f, -1.f), 100.f, materials[9]);
	*container = new Shape::Container(spheres, size);
}


__global__
void InitRandomizer(curandState* randState, int width, int height) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int index = x + y * width;
	curand_init(1994, index, 0, &randState[index]);
}

__global__
void Render(
	Image::Color* pixels, int width, int height,
	Scene::Camera* camera, Shape::Collider** collider, curandState* randState
) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;
	if (x >= width || y >= height) return;

	int index = x + y * width;
	curandState rand = randState[index];

	Image::Color color(0.f, 0.f, 0.f);
	for (int i = 0; i < 100; ++i) {
		float u = float(x + curand_uniform(&rand)) / width;
		float v = float(y + curand_uniform(&rand)) / height;
		color += ComputeColor(camera->RayAt(u, v), *collider, &rand);
	}
	color /= 100.0f;
	pixels[index] = Image::Color(sqrt(color.r), sqrt(color.g), sqrt(color.b));
}