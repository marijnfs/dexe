
all:
	nvcc -std=c++11 -arch compute_30 -I/usr/local/cuda-6.5/include/ main.cc operations.cc tensor.cc handler.cc network.cc caffe.pb.cc database.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main

debug:
	nvcc -std=c++11 -g -arch compute_30 -I/usr/local/cuda-6.5/include/ main.cc operations.cc tensor.cc handler.cc network.cc caffe.pb.cc database.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main
