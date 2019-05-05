#pragma once

#include "scene/camera.h"
#include "materials/lambert.h"
#include "materials/dielectric.h"
#include "materials/metal.h"
#include "materials/checker_texture.h"
#include "materials/diffuse_light.h"
#include "shape/container.h"
#include "shape/plane.h"

__device__
Image::Color ComputeColor(const Math::Ray& ray, Shape::Collider* collider, curandState* random, int depth = 0);

__global__
void InitColliders(Shape::Collider** container) {
	if (threadIdx.x != 0 || blockIdx.x != 0) return;	
	
	Material::Material* materials[] = {
		new Material::Dielectric(1.51f),
		new Material::Lambert(new Material::ConstantTexture(Image::Color(1.0f, 0.4f, 0.3f))),
		new Material::Lambert(new Material::ConstantTexture(Image::Color(1.0f, 0.2f, 0.3f))),
		new Material::Lambert(new Material::ConstantTexture(Image::Color(0.4f, 0.3f, 1.0f))),
		new Material::Lambert(new Material::ConstantTexture(Image::Color(0.4f, 0.4f, 0.3f))),
		new Material::Lambert(new Material::ConstantTexture(Image::Color(0.8f, 0.7f, 0.3f))),
		new Material::Dielectric(1.49f),
		new Material::Metal(Image::Color(0.9f, 0.8f, 0.6f), 0.1f),
		new Material::Metal(Image::Color(0.8f, 0.8f, 0.8f), 0.4f),
		new Material::Lambert(new Material::CheckerTexture(
			new Material::ConstantTexture(Image::Color(0.8f, 0.7f, 0.7f)),
			new Material::ConstantTexture(Image::Color(1.f, 1.f, 1.f))
		))
	};

	Material::Material* lights[] = {
		new Material::DiffuseLight(new Material::ConstantTexture(Image::Color(4.f, 3.f, 1.6f))),
		new Material::DiffuseLight(new Material::ConstantTexture(Image::Color(0.6f, 1.f, 1.f)))
	};

	Shape::Collider** colliders = new Shape::Collider*[21];

	colliders[17] = new Shape::Sphere(Math::Point(0, -1000.3f, -1.f), 1000.f, materials[9]);
	colliders[0] = new Shape::Sphere(Math::Point(0, .7f, 0.f), -.95f, materials[0]);
	colliders[18] = new Shape::Sphere(Math::Point(0, .7f, 0.f), 1.f, materials[0]);

	// Plane
	colliders[19] = new Shape::Plane(Math::Point(-2.f, 1.f, 5.f), 10.f, 10.f, lights[0]);
	colliders[20] = new Shape::Plane(Math::Point(-2.f, 1.f, -5.f), 10.f, 10.f, lights[1]);

	float coef = 2.f*3.1415925f/9.f;
	for (int i = 1; i < 9; ++i) {
		colliders[i] = new Shape::Sphere(Math::Point(2.0f*cos(i * coef), 0, 2.0f*sin(i * coef)), 0.3f, materials[i]);			
		colliders[i + 8] = new Shape::Sphere(Math::Point(3.0f*cos(i * coef + 0.2f), 0, 3.0f*sin(i * coef + 0.2f)), 0.3f, materials[9 - i]);			
	}
	
	*container = new Shape::Container(colliders, 21);
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
	for (int i = 0; i < 200; ++i) {
		float u = float(x + curand_uniform(&rand)) / width;
		float v = float(y + curand_uniform(&rand)) / height;
		color += ComputeColor(camera->RayAt(u, v, randState), *collider, &rand);
	}
	color /= 200.0f;
	pixels[index] = Image::Color(sqrt(color.r), sqrt(color.g), sqrt(color.b));
}