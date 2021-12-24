#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>

#include "util.h"

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	//=================================================
	// Reading the Dataset
	//=================================================

	std::vector<std::string> lines = read_time_machine("./data/timemachine.txt");

	//=================================================
	// Tokenization
	//=================================================

	// split words
	std::vector<std::string> tokens = tokenize(lines, "char", false);
	printf("[");
	for (int i = 0; i < 30; i++) {
		std::cout << tokens[i] << " ";
	}
	printf("]\n");

	//=================================================
	// Vocabulary
	//=================================================

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );
	auto it = counter.begin();
	printf("[");
	for(int i =0; i < 10; i++ ) {
	    // .first to access key, .second to access value
	    std::cout << it->first << ":" << it->second << " ";
	    it++;
	}
	printf("]\n");

	auto vcab = Vocab(counter, 0.0);

	auto token_to_idx = vcab.order_token_to_idx;
	printf("[");
	for(const auto& it : token_to_idx ) {
		// .first to access key, .second to access value
		std::cout << it.first << ":" << it.second << " ";
	}
	printf("]\n");

	std::cout << "vcab['t'] = " << vcab["t"] << std::endl;


	printf("[");
	for(int64_t i = 0; i < vcab.length(); i++ ) {
		std::cout << i << ":" << vcab.to_tokens({i}) << " ";
	}
	printf("]\n");

	std::cout << "corpus length: " << tokens.size() << " vocab length: " << vcab.length() << std::endl;

	std::cout << "Done!\n";
	return 0;
}


