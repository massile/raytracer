#pragma once

#include "kernels.h"

class Renderer {
private:
	// The image we want to write to
	Image::Image* image;
	// Random value generator (for the GPU)
	curandState *dRandState;
	// The camera of the scene
	Scene::Camera* dCamera;
	// The scene (contains all colliding objects)
	Shape::Collider **dContainer;

	dim3 blocks;
	dim3 threads;
public:
	Renderer(Image::Image* image, Scene::Camera* hCamera, int numThreads = 8) : 
		image(image),
		threads(numThreads, numThreads),
		blocks(image->width / numThreads + 1, image->height / numThreads + 1)
		{
			// Allocates the random value generator on the GPU
			cudaMalloc(&dRandState, image->width * image->height * sizeof(curandState));
			InitRandomizer<<<blocks, threads>>>(dRandState, image->width, image->height);
			cudaDeviceSynchronize();

			// Allocates the camera on the GPU
			cudaMallocManaged(&dCamera, sizeof(Scene::Camera));
			cudaMemcpy(dCamera, hCamera, sizeof(Scene::Camera), cudaMemcpyHostToDevice);
			cudaMalloc(&dContainer, sizeof(Shape::Collider*));
			
			// Allocates the scene on the GPU
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

/**
 * Computes the final color of a ray arriving at a pixel
 * It uses a recursive algorithm
 * 
 * @param ray - The ray casted from the light
 * @param collider - The object containing all physical entities that can collide with a ray of light
 * @param random - A pointer to a random number generator (provided by CUDA)
 * @param depth - How many times the ray has collided with an object
 */
__device__
Image::Color ComputeColor(const Math::Ray& ray, Shape::Collider* collider, curandState* random, int depth) {
	Shape::Interface interface;
	// If the ray hits an object in the scene
	if (collider->Hit(ray, interface, 0.001f, 10000.0f)) {
		Image::Color color;
		Math::Ray out;
		// Calculates the rays emitted by a light source (black if it's not emitting anything)
		Image::Color emitted = interface.material->Emitted(interface.point);
		// Calculates how the ray of light should be scattered (i.e. reflected / refracted)
		// The new ray is stored in the `out` variable
		bool shouldScatter = interface.material->Scatter(ray, interface, color, out, random);
		// If the light is not scattered, the recursion ends
		if (depth == 40 || !shouldScatter) {
			return emitted;
		}
		// We calculate the new color of the ray using the same algorithm on the scattered ray 
		return color * ComputeColor(out, collider, random, depth + 1);
	}

	// Ambient light (when the light comes from the sky)
	return Image::Color(0.1f, 0.1f, 0.15f);
}