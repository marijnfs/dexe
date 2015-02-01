all:
	nvcc -std=c++11 -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main

debug:
	nvcc -std=c++11 -g -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main
