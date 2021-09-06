

#include <iostream>
#include <torch/extension.h>
#include <vector>


std::vector<torch::Tensor> voxelize(torch::Tensor cloud,
	torch::Tensor size,
	torch::optional<torch::Tensor> optional_start,
	torch::optional<torch::Tensor> optional_end,
	torch::optional<bool> optional_return_centers) {

	torch::Tensor start;
	torch::Tensor end;

	if (optional_start.has_value()) {
		start = optional_start.value();
	}

	else {
		start = cloud.min();
	}
	if (optional_end.has_value()) {
		end = optional_end.value();
	}

	else {
		end = cloud.max();
	}

	auto return_centers = optional_return_centers.value();

	torch::Tensor centers;
	torch::Tensor n_voxels = ((end - start) / size).floor();
	std::vector<torch::Tensor> out;
	int n_dim = size.numel();

	size = size.view({ 1,-1 });
	start = start.view({ 1,-1 });



	cloud -= start;
	cloud = cloud.true_divide(size);
	if (return_centers) {
		centers = cloud * size + start + size / 2;
	}

	n_voxels = n_voxels.cumprod(0);
	n_voxels =
		torch::cat({ torch::ones(1, n_voxels.options()), n_voxels }, 0);
	n_voxels = n_voxels.narrow(0, 0, n_dim);
	cloud *= n_voxels.view({ 1,-1 });
	cloud = cloud.floor();
	torch::Tensor indexes = cloud.sum(1).unsqueeze(-1);


	out.push_back(indexes.to(torch::kInt));
	if (return_centers)
	{
		out.push_back(centers);

	}
	
	return out;
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxelize", &voxelize, "Voxelize function");
}