///Tensor2D

template <typename T>
Tensor2D<T>::Tensor2D(int w_, int h_): 
  w(w_), h(h_), allocated(true)
{
	handle_error( cudaMalloc( (void**)&data, sizeof(T) * h * w));
	if (ZERO_ON_INIT)
	  zero();
}

template <typename T>
Tensor2D<T>::Tensor2D(int w_, int h_, T *data_): 
  w(w_), h(h_), allocated(false), data(data_)
{
}

template <typename T>
Tensor2D<T>::~Tensor2D() {
   	if (allocated)
	  cudaFree(data);
}

template <typename T>
void Tensor2D<T>::zero() {
  handle_error( cudaMemset(data, 0, sizeof(T) * w * h));
}

template <typename T>
vector<T> Tensor2D<T>::to_vector() {
	vector<T> vec(h * w);
	handle_error( cudaMemcpy(&vec[0], data, vec.size() * sizeof(T), cudaMemcpyDeviceToHost));
	return vec;
}

template <typename T>
void Tensor2D<T>::from_vector(vector<T> &in) {
  handle_error( cudaMemcpy(data, &in[0], in.size() * sizeof(T), cudaMemcpyHostToDevice));
}
