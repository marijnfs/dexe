all:
	nvcc -std=c++11 -O3 -L. -Iinc -arch compute_30 -I/usr/local/cuda/include/ -lmcdnn -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options main.cc -o main

debug:
	nvcc -std=c++11 -g -Iinc -arch compute_30 -I/usr/local/cuda/include/ src/*.cc main.cc -lcudnn -lcurand -lcublas -lleveldb -lprotobuf -lopencv_core -lopencv_highgui -lopencv_imgproc -lboost_program_options -o main

lib:
	nvcc -std=c++11 -O3 -Iinc -shared --compiler-options '-fPIC' -arch compute_30 src/*.cc src/*.cu -o libmcdnn.so
	#cp libmcdnn.so ../selfgo/
	#cp inc/*.h ../selfgo/

debuglib:
	nvcc -std=c++11 -g -Iinc -shared --compiler-options '-fPIC' -arch compute_30 src/*.cc src/*.cu -o libmcdnn.so

install:
	sudo cp libmcdnn.so /usr/local/lib


libdebug:
	nvcc -std=c++11 -g -Iinc -shared --compiler-options '-fPIC' -arch compute_30 src/*.cc src/*.cu -o libmcdnn.so
