#include "ch_20_util.h"

torch::Tensor distance_matrix(torch::Tensor x, torch::Tensor y) {
	assert(x.size(1) == y.size(1));
    int m = x.size(0);
    int n = y.size(0);
	torch::Tensor z = torch::zeros({m, n});
    for(auto& i : range(m, 0) ) {
        for(auto& j : range(n, 0)) {
            z[i][j] = std::sqrt(torch::sum(torch::pow((x[i] - y[j]),2)).data().item<float>());
        }
    }
    return z;
}

torch::Tensor rbfkernel(torch::Tensor x1, torch::Tensor x2, float ls) {
	torch::Tensor dist = distance_matrix(x1.unsqueeze(1), x2.unsqueeze(1));
    return torch::exp(-(1. / ls / 2) * (torch::pow(dist, 2)));
}

std::vector<double> tensorTovec(torch::Tensor A) {
	std::vector<double> v;
	for(int j = 1; j < A.size(0); j++)
		v.push_back(A[j].data().item<double>());

	return v;
}



