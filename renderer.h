#pragma once

#include "kernels.h"

class Renderer {
private:
	Image::Image* image;
	Scene::Camera* dCamera;
	Shape::Collider **dContainer, **dSpheres;
	Material::Material **dMaterials;

	curandState *dRandState;

	dim3 blocks;
	dim3 threads;
public:
	Renderer(Image::Image* image, Scene::Camera* hCamera, int numThreads = 8) : 
		image(image),
		threads(numThreads, numThreads),
		blocks(image->width / numThreads + 1, image->height / numThreads + 1)
		{
			cudaMallocManaged(&dCamera, sizeof(Scene::Camera));
			cudaMemcpy(dCamera, hCamera, sizeof(Scene::Camera), cudaMemcpyHostToDevice);
			
			int shapeSize = 10;
			int materialSize = 2;
			cudaMalloc(&dContainer, sizeof(Shape::Collider*));
			cudaMalloc(&dSpheres, shapeSize * sizeof(Shape::Collider*));
			cudaMalloc(&dMaterials, materialSize * sizeof(Material::Lambert*));
			InitColliders<<<1, 1>>>(dContainer, dSpheres, dMaterials, shapeSize);
			cudaDeviceSynchronize();

			cudaMalloc(&dRandState, image->width * image->height * sizeof(curandState));
			InitRandomizer<<<blocks, threads>>>(dRandState, image->width, image->height);
			cudaDeviceSynchronize();
		}

	~Renderer() {
		cudaFree(dMaterials);
		cudaFree(dSpheres);
		cudaFree(dContainer);
		cudaFree(dCamera);
	}

	void Run(const char* fileName) {
		Render<<<blocks, threads>>>(image->pixels, image->width, image->height, dCamera, dContainer, dRandState);
		cudaDeviceSynchronize();
		image->Write(fileName);
	}
};

__device__
Image::Color ComputeColor(const Math::Ray& ray, Shape::Collider* collider, curandState* random, int depth) {
	if (depth == 50) {
		return Image::Color(0.f, 0.f, 0.f);
	}

	Shape::Interface interface;
	if (collider->Hit(ray, interface, 0.001f, 10000.0f)) {
		Image::Color color;
		Math::Ray out;
		interface.material->Scatter(interface, color, out, random);
		return color * ComputeColor(out, collider, random, depth + 1);
	}

	Math::Vector normDir = Math::Normalize(ray.direction);
	float t = 0.5f * normDir.y + 0.5f;
	return Math::Lerp(t, Image::Color(1.0f, 1.0f, 1.0f), Image::Color(0.6f, 0.7f, 0.9f));
}