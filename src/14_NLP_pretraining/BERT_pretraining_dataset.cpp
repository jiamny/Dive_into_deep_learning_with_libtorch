
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	torch::Device device(torch::kCPU);
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> paragraphs = _read_wiki(data_dir, 1000);

	std::cout << paragraphs.size() << '\n';
	std::vector<std::vector<std::string>> lines = paragraphs[0];
	std::cout << lines.size() << '\n';
	std::vector<std::string> tks = lines[0];
	std::cout << tks.size() << '\n';
	printVector(tks);

	int64_t max_len = 64;

	_WikiTextDataset train_set(paragraphs, max_len);

	std::cout << train_set.size() << '\n';

	auto rlt = train_set.get(0);
	std::cout << "all_token_ids:\n" << std::get<0>(rlt) << '\n';
	std::cout << "all_segments:\n"  <<  std::get<1>(rlt) << '\n';
	std::cout << "valid_lens:\n"    << std::get<2>(rlt) << '\n';
	std::cout << "all_pred_positions:\n" << std::get<3>(rlt) << '\n';
	std::cout << "all_mlm_weights:\n"    << std::get<4>(rlt) << '\n';
	std::cout << "all_mlm_labels:\n"     << std::get<5>(rlt) << '\n';
	std::cout << "nsp_labels:\n"         << std::get<6>(rlt) << '\n';
	std::cout << train_set.vocab.length() << '\n';

	std::cout << "Done!\n";
}




