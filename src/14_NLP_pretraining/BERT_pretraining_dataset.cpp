
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
	/*
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';
	*/

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> paragraphs = _read_wiki(data_dir, 1000);

	std::cout << paragraphs.size() << '\n';
	std::vector<std::vector<std::string>> lines = paragraphs[0];
	std::cout << lines.size() << '\n';
	std::vector<std::string> tks = lines[0];
	std::cout << tks.size() << '\n';
	printVector(tks);

	int64_t max_len = 64, batch_size = 512;

	_WikiTextDataset train_set(paragraphs, max_len);

	std::cout << train_set.size().value() << '\n';

	auto rlt = train_set.get(0);
	std::cout << "all_token_ids:\n" << rlt.data << '\n';
	std::cout << "tidxs:\n"  <<  rlt.target << '\n';

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
			   torch::Tensor, torch::Tensor> dts = train_set.getData();


	const torch::Tensor all_segments = std::get<0>(dts), valid_lens = std::get<1>(dts),
				  all_pred_positions = std::get<2>(dts), all_mlm_weights = std::get<3>(dts),
				  all_mlm_labels = std::get<4>(dts), nsp_labels = std::get<5>(dts);

	std::cout << "all_segments:\n"  <<  all_segments.sizes() << '\n';
	std::cout << "valid_lens:\n"    << valid_lens.sizes() << '\n';
	std::cout << "all_pred_positions:\n" << all_pred_positions.sizes() << '\n';
	std::cout << "all_mlm_weights:\n"    << all_mlm_weights.sizes() << '\n';
	std::cout << "all_mlm_labels:\n"     << all_mlm_labels.sizes() << '\n';
	std::cout << "nsp_labels:\n"         << nsp_labels.sizes() << '\n';
	std::cout << train_set.getVocab().length() << '\n';

	auto dataset = train_set.map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
				        	std::move(dataset),
							torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	for(auto& dt : *train_iter ) {
		    torch::Tensor data = dt.data;
		    const torch::Tensor target = dt.target;
		    std::cout << "data.sizes: " << data.sizes() << '\n';
		    std::cout << "target.sizes: " << target.sizes() << '\n';
		    std::cout << "target: " << target << '\n';

		    auto slt_segments = torch::index_select(all_segments, 0, target.squeeze());
		    std::cout << "slt_segments.sizes: " << slt_segments.sizes() << '\n';
		    break;
	}


	std::cout << "Done!\n";
	return 0;
}




