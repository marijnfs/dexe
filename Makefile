all:
	nvcc -arch compute_30 -I/usr/local/cuda-6.5/include/ main.cc layers.cc tensor.cc handler.cc caffe.pb.cc database.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main

debug:
	nvcc -g -arch compute_30 -I/usr/local/cuda-6.5/include/ main.cc layers.cc tensor.cc handler.cc caffe.pb.cc database.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -o main
