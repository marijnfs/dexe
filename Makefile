all:
	nvcc -std=c++11 -O3 -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -o main

debug:
	nvcc -std=c++11 -g -arch compute_30 -I/usr/local/cuda-6.5/include/ *.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -o main

lib:
	nvcc -std=c++11 -shared --compiler-options '-fPIC' -arch compute_30 adv.cc balancer.cc caffe.pb.cc database.cc handler.cc img.cc loss.cc network.cc operations.cc options.cc tensor.cc -o libmcdnn.so

