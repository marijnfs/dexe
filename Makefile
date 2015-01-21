all:
	nvcc -arch compute_30 -I/usr/local/cuda-6.5/include/ main.cc layers.cc tensor.cc handler.cc -lcudnn -lcurand -lcublas -o main

debug:
	nvcc -arch compute_30 -g -I/usr/local/cuda-6.5/include/ main.cc layers.cc tensor.cc handler.cc -lcudnn -lcurand -lcublas -o main
