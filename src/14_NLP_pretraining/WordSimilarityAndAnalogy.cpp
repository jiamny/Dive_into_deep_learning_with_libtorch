/*
 * WordSimilarityAndAnalogy.cpp
 *
 */
#include <unistd.h>
#include <iomanip>
#include <cmath>
#include <torch/utils.h>
#include <torch/torch.h>
#include "../ProgressBar.hpp"
#include "../utils/ch_14_util.h"


struct TokenEmbedding {
	std::unordered_map<std::string, int64_t> token_to_idx;
	std::vector<std::string> idx_to_token;
	torch::Tensor idx_to_vec;
	int64_t unknown_idx;
	TokenEmbedding(std::string embedding_name, int64_t num_read ) {
		std::pair<std::vector<std::string>, torch::Tensor> dt = _load_embedding(
	            embedding_name, num_read);
		idx_to_token = dt.first, idx_to_vec = dt.second;

		unknown_idx = 0;
		for(int i = 0; i < idx_to_token.size(); i++) {
			std::string str = idx_to_token[i];
			token_to_idx[str] = i;
		}
	}

	std::pair<std::vector<std::string>, torch::Tensor> _load_embedding(std::string embedding_name, int64_t num ) {
		std::vector<std::string> idx_to_token = {"<unk>"};
		std::vector<float> idx_to_vec;
		std::vector<torch::Tensor> t_vec;

		std::string f = embedding_name + "/vec.txt";
	    std::string line;
	    const std::string WHITESPACE = " ";
	    std::ifstream fL(f.c_str());
	    int64_t cnt = 1;

	    // initialize the bar
	    progresscpp::ProgressBar progressBar(num, 70);

	    if( fL.is_open() ) {
	    	while ( std::getline(fL, line) ) {

	    	    line = std::regex_replace(line, std::regex("\\\n"), "");
	    	    line = strip(line);

	    	    size_t pos = 0;

	    		if( line.length() > 1 ) {
	    			std::vector<float> tks{};
	    			size_t ps = 0;
	    			bool first_tk = true;
	    			while ((ps = line.find(WHITESPACE)) != std::string::npos) {
	    			    std::string tk = line.substr(0, ps);
	    			    // trim space
	    			    tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
	    			    tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
	    			    if( tk.length() > 0 ) {
	    			    	if( first_tk ) {
	    			    		idx_to_token.push_back(tk);
	    			    		first_tk = false;
	    			    	} else {
	    			    		tks.push_back(std::atof(tk.c_str()));
	    			    	}
	    			    }
	    			    line.erase(0, ps + WHITESPACE.length());
	    			}

	    			std::string tk = line.substr(0, ps);
	    			// trim space
	    			tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
	    			tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
	    			if( tk.length() > 0 ) {
	    				tks.push_back(std::atof(tk.c_str()));
	    			}

	    			if( t_vec.size() < 1 ) {
	    				std::vector<float> ff{};
	    				for( int64_t i = 0; i < static_cast<int64_t>(tks.size()); i++)
	    					ff.push_back(0.0);
	    				torch::Tensor ft = torch::from_blob(tks.data(), {1, static_cast<int64_t>(tks.size())}).clone();
	    			    t_vec.push_back(ft);
	    			}
	    			torch::Tensor t = torch::from_blob(tks.data(), {1, static_cast<int64_t>(tks.size())}).clone();
	    			t_vec.push_back(t);

	    		    cnt++;
	    		}

	    		if( num > 0 ) {
	    			++progressBar;
	    			// display the bar
	    			progressBar.display();
	    		}

	    		if( num > 0 && cnt >= num )
	    			break;
	    	}
	    }
	    fL.close();

	    torch::Tensor Ts = torch::cat(t_vec, 0);
	    return std::make_pair(idx_to_token, Ts);
	}

    torch::Tensor operator[](std::vector<std::string> tokens) {
    	torch::Tensor vecs;
    	std::vector<int64_t> indices;
    	for(auto& token : tokens) {
    		indices.push_back(static_cast<int64_t>(token_to_idx[token]));
    	}

        vecs = idx_to_vec.index({torch::tensor(indices).reshape({-1,1}), Slice()});
        return vecs;
    }

    int64_t __len__(void) {
        return idx_to_token.size();
    }
};


std::pair<torch::Tensor, std::vector<torch::Tensor>> knn(torch::Tensor W, torch::Tensor x, int64_t k) {
	std::vector<torch::Tensor> Cs;
    // Add 1e-9 for numerical stability
    torch::Tensor cos1 = torch::mv(W, x.reshape({-1,}));
    torch::Tensor t = torch::sum(W * W, {1});
    torch::Tensor cos2 = torch::sqrt(t.add(1e-9)) *	torch::sqrt((x * x).sum());
    torch::Tensor cos = cos1.div(cos2);

    torch::Tensor _, topk;
    std::tie(_, topk) = torch::topk(cos, k);

    for(int64_t i = 0; i < k; i++) {
    	int64_t s = topk[i].data().item<int64_t>();
    	Cs.push_back(cos[s].clone());
    }
    return std::make_pair(topk, Cs);
}

void get_similar_tokens(std::vector<std::string> query_token, int64_t k, TokenEmbedding embed) {
	torch::Tensor topk;
	std::vector<torch::Tensor> cos;

    std::tie(topk, cos) = knn(embed.idx_to_vec, embed[query_token], k + 1);

    for(int j = 1; j < cos.size(); j++) {
    	int64_t i = topk[j].data().item<int64_t>();
    	printf("cosine sim=%.3f %s\n", cos[j].data().item<float>(), embed.idx_to_token[i].c_str());
    }
}

std::string get_analogy(std::string token_a, std::string token_b, std::string token_c,
		TokenEmbedding embed) {
	std::vector<std::string> tokens;
	tokens.push_back(token_a);
	tokens.push_back(token_b);
	tokens.push_back(token_c);
    torch::Tensor vecs = embed[tokens];
    torch::Tensor x = vecs[1] - vecs[0] + vecs[2];

	torch::Tensor topk;
	std::vector<torch::Tensor> cos;
    std::tie(topk, cos) = knn(embed.idx_to_vec, x, 1);
    int64_t s = topk[0].data().item<int64_t>();
    return embed.idx_to_token[s];  // Remove unknown words
}


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	//auto cuda_available = torch::cuda::is_available();
	//torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	//std::cout << (cuda_available ? "CUDA available. Running on GPU." : "Running on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string embedding_name = "/media/hhj/localssd/PycharmProjects/d2l-en/d2l-pytorch-sagemaker/data/glove.6B.50d";
	int64_t num_read = 10000;

	std::cout << "TokenEmbedding...\n";
	auto glove_6b50d = TokenEmbedding(embedding_name, num_read );

	std::cout << "\nglove_6b50d.token_to_idx['<unk>']: " << glove_6b50d.token_to_idx["<unk>"]
				  << " glove_6b50d.idx_to_token[0]: " << glove_6b50d.idx_to_token[0] << '\n';

	std::cout << "glove_6b50d.token_to_idx['beautiful']: " << glove_6b50d.token_to_idx["beautiful"]
			  << " glove_6b50d.idx_to_token[3367]: " << glove_6b50d.idx_to_token[3367] << '\n';

	// Applying Pretrained Word Vectors
	// Word Similarity
	std::cout << "\nWord Similarity:\n";
	std::vector<std::string> embed = {"chip"};
	get_similar_tokens(embed, 3, glove_6b50d);
	get_similar_tokens({"baby"}, 3, glove_6b50d);
	get_similar_tokens({"beautiful"}, 3, glove_6b50d);

	// Word Analogy
	std::cout << "\nWord Analogy:\n";
	printf("%s\n", get_analogy("man", "woman", "son", glove_6b50d).c_str());
	printf("%s\n", get_analogy("beijing", "china", "tokyo", glove_6b50d).c_str());
	printf("%s\n", get_analogy("bad", "worst", "big", glove_6b50d).c_str());
	printf("%s\n", get_analogy("do", "did", "go", glove_6b50d).c_str());

	std::cout << "Done!\n";

	return 0;
}



