#include <sstream>
#include "adv.h"

using namespace std;


void MakeAdvDatabase(Database &in, Database &out, Network<float> &network, float step) {
	Timer t;
	cout << "Making adv database" << endl;

	Tensor<float> &x_grad(network.tensors[0]->grad);

	Tensor<float> x(network.shapes[0]);
	Tensor<float> g1(network.shapes[0]);
	Tensor<float> g2(network.shapes[0]);
	Tensor<float> g3(network.shapes[0]);
	Tensor<float> g4(network.shapes[0]);

	for (size_t i(0); i < in.N; ++i) {
		caffe::Datum datum = in.get_image(i);
		const float *img_data = datum.float_data().data();
		network.forward(img_data);
		network.calculate_loss(datum.label());
		network.backward_data();
		float g1_norm = x_grad.norm();

		x.from_tensor(network.tensors[0]->x); //backup x
		g1.from_tensor(network.tensors[0]->grad);
	
		if (i % 1000 == 0) {
			ostringstream oss;
			oss << "/home/marijnfs/tmp/" << i << "_norm.bmp";
			x.write_img(oss.str());
			cout << "loss: " << network.loss() << " ";
		}

		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.5 * step / g1_norm);
		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();
		float g2_norm = x_grad.norm();
		g2.from_tensor(network.tensors[0]->grad);
	
		network.tensors[0]->x.from_tensor(x);
		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.5 * step / g2_norm);

		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();
		float g3_norm = x_grad.norm();
		g3.from_tensor(network.tensors[0]->grad);
		
		network.tensors[0]->x.from_tensor(x);
		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -1.0 * step / g3_norm);
		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();

		float g4_norm = x_grad.norm();
		g4.from_tensor(network.tensors[0]->grad);
		if (i % 1000 == 0)
			cout << network.loss() << endl;
		add_cuda<float>(g1.data, x.data, x.size(), -step / 6.0 / g1_norm);
		add_cuda<float>(g2.data, x.data, x.size(), -step * 2.0 / 6.0 / g2_norm);
		add_cuda<float>(g3.data, x.data, x.size(), -step * 2.0 / 6.0 / g3_norm);
		add_cuda<float>(g4.data, x.data, x.size(), -step / 6.0 / g4_norm);
		//x.write_img("adv_rk4.bmp");
		//throw "";

		if (i % 1000 == 0) {
			ostringstream oss;
			oss << "/home/marijnfs/tmp/" << i << "_adv.bmp";
			x.write_img(oss.str());
		}

		vector<float> adv_img = x.to_vector();
		normalize(&adv_img);

		datum.clear_float_data();
		for (size_t i(0); i < adv_img.size(); ++i)
			datum.add_float_data(adv_img[i]);
		string output;
		datum.SerializeToString(&output);
		out.db->Put(leveldb::WriteOptions(), out.get_key(i), output);
	}
	cout << "took: " << t.since() << endl;
}

void AddNAdv(Database &in, Database &out, Network<float> &network, int n, float percent) {
	Timer t;
	cout << "Adding " << n << "random adv to database" << endl;

	Tensor<float> &x_grad(network.tensors[0]->grad);

	Tensor<float> x(network.shapes[0]);
	Tensor<float> g1(network.shapes[0]);
	Tensor<float> g2(network.shapes[0]);
	Tensor<float> g3(network.shapes[0]);
	Tensor<float> g4(network.shapes[0]);

	Indices indices(in.N);
	indices.shuffle();
	for (size_t i(0); i < n; ++i) {
		caffe::Datum datum = in.get_image(indices[i]);
		const float *img_data = datum.float_data().data();
		network.forward(img_data);
		network.calculate_loss(datum.label());
		network.backward_data();
		float step = network.loss() * percent;

		float g1_norm = x_grad.norm();
		g1_norm *= g1_norm;
		x.from_tensor(network.tensors[0]->x); //backup x
		g1.from_tensor(network.tensors[0]->grad);
	
		if (i % 200 == 0) {
			ostringstream oss;
			oss << "/home/marijnfs/tmp/" << i << "_norm.bmp";
			x.write_img(oss.str());
			cout << "loss: " << network.loss() << " ";
		}

		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.5 * step / g1_norm);
		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();
		float g2_norm = x_grad.norm();
		g2_norm *= g2_norm;
		g2.from_tensor(network.tensors[0]->grad);
	
		network.tensors[0]->x.from_tensor(x);
		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -.5 * step / g2_norm);

		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();
		float g3_norm = x_grad.norm();
		g3_norm *= g3_norm;
		g3.from_tensor(network.tensors[0]->grad);
		
		network.tensors[0]->x.from_tensor(x);
		add_cuda<float>(x_grad.data, network.tensors[0]->x.data, network.tensors[0]->x.size(), -1.0 * step / g3_norm);
		network.forward();
		network.calculate_loss(datum.label());
		network.backward_data();

		float g4_norm = x_grad.norm();
		g4_norm *= g4_norm;
		g4.from_tensor(network.tensors[0]->grad);

		if (i % 200 == 0)
			cout << network.loss() << endl;

		add_cuda<float>(g1.data, x.data, x.size(), -step / 6.0 / g1_norm);
		add_cuda<float>(g2.data, x.data, x.size(), -step * 2.0 / 6.0 / g2_norm);
		add_cuda<float>(g3.data, x.data, x.size(), -step * 2.0 / 6.0 / g3_norm);
		add_cuda<float>(g4.data, x.data, x.size(), -step / 6.0 / g4_norm);
		//x.write_img("adv_rk4.bmp");
		//throw "";

		if (i % 200 == 0) {
			ostringstream oss;
			oss << "/home/marijnfs/tmp/" << i << "_adv.bmp";
			x.write_img(oss.str());
		}

		vector<float> adv_img = x.to_vector();
		normalize(&adv_img);

		datum.clear_float_data();
		for (size_t i(0); i < adv_img.size(); ++i)
			datum.add_float_data(adv_img[i]);

		out.add(datum);
	}
	cout << "took: " << t.since() << endl;
}
