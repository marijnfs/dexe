all:
	nvcc -std=c++11 -O3 -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -o main

debug:
	nvcc -std=c++11 -g -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -o main
