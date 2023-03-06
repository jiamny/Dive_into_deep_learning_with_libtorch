#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"

class _WikiTextDataset {
public:
	_WikiTextDataset(std::vector<std::vector<std::vector<std::string>>> paragraphs, int64_t max_len) {
        // Input `paragraphs[i]` is a list of sentence strings representing a
        // paragraph; while output `paragraphs[i]` is a list of sentences
        // representing a paragraph, where each sentence is a list of tokens
		//std::vector<std::vector<std::string>> setences;
		std::vector<std::string> tokens;

		for(auto& paragraph : paragraphs) {
			for(auto& setence : paragraph) {

				for(auto& tk : setence)
					tokens.push_back(tk);

				//setences.push_back(setence);
			}
		}
//		std::cout << "tokens: " << tokens.size() << '\n';
//		std::cout << "tokens[0]: " << tokens[0] << '\n';

		std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

		std::vector<std::string> reserved_tokens;
		reserved_tokens.push_back("<pad>");
		reserved_tokens.push_back("<mask>");
		reserved_tokens.push_back("<cls>");
		reserved_tokens.push_back("<sep>");
		auto vocab = Vocab(counter, 5.0, reserved_tokens);

//		std::cout << "the: " << vocab["the"] << "\n";
//		std::cout << "counter: " << counter[0].second << "\n";

	    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
									 std::vector<int64_t>, std::vector<int64_t>, bool>> examples;
	    for( auto& paragraph : paragraphs ) {
	    	// Get data for the next sentence prediction task
//	    	std::cout << "paragraph: " << paragraph.size() << "\n";
//	    	printVector(paragraph);

	    	auto nsp_data = _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len);

//	    	std::cout << "nsp_data: " << nsp_data.size() << "\n";

	    	for( auto& nsp : nsp_data ) {
	    		std::vector<std::string> tks;
	    		std::vector<int64_t>     segments;
	    		bool is_next;
	    		tks = std::get<0>(nsp);
				segments = std::get<1>(nsp);
				is_next = std::get<2>(nsp);
	    		// Get data for the masked language model task
	    		std::vector<int64_t> token_ids, pred_positions, mlm_pred_label_ids;
	    		std::tie(token_ids, pred_positions, mlm_pred_label_ids) = _get_mlm_data_from_tokens(tks, vocab);

	    		examples.push_back(std::make_tuple(token_ids, pred_positions, mlm_pred_label_ids, segments, is_next));
	    	}
	    }
//	    std::cout << "examples: " << examples.size() << "\n";

        // Pad inputs
		std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
				   std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
		rlt = _pad_bert_inputs(examples, max_len, vocab);
        all_token_ids = std::get<0>(rlt);
        all_segments = std::get<1>(rlt);
		valid_lens = std::get<2>(rlt);
        all_pred_positions = std::get<3>(rlt);
		all_mlm_weights = std::get<4>(rlt);
        all_mlm_labels = std::get<5>(rlt);
		nsp_labels = std::get<6>(rlt);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> get(int64_t idx) {
        return std::make_tuple(all_token_ids[idx], all_segments[idx],
                valid_lens[idx], all_pred_positions[idx],
                all_mlm_weights[idx], all_mlm_labels[idx],
                nsp_labels[idx]);
	}

	int64_t size() {
        return all_token_ids.size();
	}
private:
	Vocab vocab;
	std::vector<torch::Tensor> all_token_ids, all_segments, valid_lens,
							   all_pred_positions, all_mlm_weights, all_mlm_labels, nsp_labels;
};

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> paragraphs = _read_wiki(data_dir, 100);

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

	std::cout << "Done!\n";
}




