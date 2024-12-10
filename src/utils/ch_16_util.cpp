#include "ch_16_util.h"

std::string strip( const std::string& s ) {
	const std::string WHITESPACE = " \n\r\t\f\v";

	size_t start = s.find_first_not_of(WHITESPACE);
	std::string ls = (start == std::string::npos) ? "" : s.substr(start);

	size_t end = ls.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : ls.substr(0, end + 1);
}

std::vector<std::string> stringSplit(const std::string& str, char delim) {
    std::stringstream ss(str);
    std::string item;
    std::vector<std::string> elems;
    while (std::getline(ss, item, delim)) {
    	item = strip(item);
        if(! item.empty()) {
            elems.push_back(item);
        }
    }
    return elems;
}

torch::Tensor RangeTensorIndex(int64_t num, bool suffle) {
	std::vector<int64_t> idx;
	for( int64_t i = 0; i < num; i++ )
		idx.push_back(i);

	if( suffle ) {
		auto seed = std::chrono::system_clock::now().time_since_epoch().count();
		std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
	}

	torch::Tensor RngIdx = (torch::from_blob(idx.data(), {num}, at::TensorOptions(torch::kInt64))).clone();
	return RngIdx;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> train_test_split(torch::Tensor X,
		torch::Tensor y, double test_size, bool suffle) {
	torch::Tensor x_train, x_test, y_train, y_test;
	int num_records = X.size(0);
	if( suffle ) {
		torch::Tensor sidx = RangeTensorIndex(num_records, suffle);
		X = torch::index_select(X, 0, sidx.squeeze());
		y = torch::index_select(y, 0, sidx.squeeze());
	}

	// ---- split train and test datasets
	int train_size = static_cast<int>(num_records * (1 - test_size));
	x_train = X.index({Slice(0, train_size), Slice()});
	x_test = X.index({Slice(train_size, None), Slice()});
	y_train = y.index({Slice(0, train_size), Slice()});
	y_test = y.index({Slice(train_size, None), Slice()});
	return std::make_tuple(x_train, x_test, y_train, y_test);
}




