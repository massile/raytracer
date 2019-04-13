COMPILER = "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx86\x64"
NVCC = nvcc -ccbin=$(COMPILER)

raytracer: main.cu
	$(NVCC) main.cu -o bin/raytracer

out: raytracer
	./bin/raytracer