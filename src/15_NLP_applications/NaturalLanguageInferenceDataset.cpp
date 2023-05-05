#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_15_util.h"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);

	torch::manual_seed(123);

	//Read the SNLI dataset into premises, hypotheses, and labels.

	const std::string data_dir = "./data/snli_1.0";
	const bool is_train = true;

	auto train_data = read_snli(data_dir, is_train, 0);

	for(int c = 0; c < 20; c++) {
		std::cout << std::get<0>(train_data)[c] << '\n';
		std::cout << std::get<1>(train_data)[c] << '\n';
		std::cout << std::get<2>(train_data)[c] << '\n';
	}

	auto test_data = read_snli(data_dir, false, 0);
	for(auto& data : {train_data, test_data}) {
		std::vector<int64_t> row = std::get<2>(data);
		std::cout << "[";
		for(int i = 0; i < 3; i++) {
			if( i < 2 )
				std::cout << std::count(row.begin(), row.end(), i) << ", ";
			else
				std::cout << std::count(row.begin(), row.end(), i);
		}
		std::cout << "]\n";
	    //print([[row for row in data[2]].count(i) for i in range(3)])
	}

	float min_freq = 5.0f;
	std::vector<std::string> reserved_tokens;
	reserved_tokens.push_back("<pad>");

	Vocab vocab;
	vocab = get_snil_vocab( train_data, min_freq, reserved_tokens);
	size_t num_steps = 50;
	size_t batch_size =128;
	std::cout << "vocab.length: " << vocab.length() << '\n';

	auto dataset = SNLIDataset(train_data, num_steps, vocab).map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        												std::move(dataset), batch_size);

	for(auto& dt : *train_iter ) {
	    auto data = dt.data;
	    auto target = dt.target;
	    std::cout << "data.sizes: " << data.sizes() << '\n';
	    std::cout << "target.sizes: " << target.sizes() << '\n';
	    std::cout << "data[:,0:50]: " << data.index({Slice(), Slice(0, num_steps)}).sizes() << '\n';
	    std::cout << "data[:,50:100]: " << data.index({Slice(), Slice(num_steps, 2*num_steps)}).sizes() << '\n';
	    break;
	}

	std::cout << "Done!\n";
}




