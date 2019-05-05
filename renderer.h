#pragma once

#include "kernels.h"

class Renderer {
private:
	Image::Image* image;

	curandState *dRandState;
	Scene::Camera* dCamera;
	Shape::Collider **dContainer;

	dim3 blocks;
	dim3 threads;
public:
	Renderer(Image::Image* image, Scene::Camera* hCamera, int numThreads = 8) : 
		image(image),
		threads(numThreads, numThreads),
		blocks(image->width / numThreads + 1, image->height / numThreads + 1)
		{
			cudaMalloc(&dRandState, image->width * image->height * sizeof(curandState));
			InitRandomizer<<<blocks, threads>>>(dRandState, image->width, image->height);
			cudaDeviceSynchronize();

			cudaMallocManaged(&dCamera, sizeof(Scene::Camera));
			cudaMemcpy(dCamera, hCamera, sizeof(Scene::Camera), cudaMemcpyHostToDevice);
			cudaMalloc(&dContainer, sizeof(Shape::Collider*));
			
			InitColliders<<<1, 1>>>(dContainer);
			cudaDeviceSynchronize();
		}

	~Renderer() {
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
	Shape::Interface interface;
	if (collider->Hit(ray, interface, 0.001f, 10000.0f)) {
		Image::Color color;
		Math::Ray out;
		Image::Color emitted = interface.material->Emitted(interface.point);
		bool shouldScatter = interface.material->Scatter(ray, interface, color, out, random);
		if (depth == 40 || !shouldScatter) {
			return emitted;
		}
		return color * ComputeColor(out, collider, random, depth + 1);
	}

	return Image::Color(0.1f, 0.1f, 0.15f);
}