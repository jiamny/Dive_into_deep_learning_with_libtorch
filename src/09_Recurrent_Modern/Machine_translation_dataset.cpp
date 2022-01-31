#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <vector>

#include <regex>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <random>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

#include "../matplotlibcpp.h"
namespace plt = matplotlibcpp;


std::string read_data_nmt(const std::string filename) {

	std::string raw_test = "";
	std::string line;
	std::ifstream fs;

	fs.open(filename.c_str()); 		// OPen the file
	if( fs.is_open() ) {      		// Fail bit will be set if file does not exist
		while( ! fs.eof() ) {
			getline(fs, line);
			raw_test += (line + "\n");
		}
		//fs.open(filename, std::ios_base::in | std::ios_base::out);
	} else {
		fs.close();
		std::cout << "Error opening file\n";
		return "";
	}

	fs.close();
	return raw_test;
}

bool no_space(char c, char pc) {
	std::set<char> s = {',', '.', '!', '?'};

	return ((s.find(c) != s.end()) && pc != ' ');
}


std::string preprocess_nmt( std::string raw_test) {
    // Preprocess the English-French dataset."""
	std::string processed = "";
	for( std::string::size_type i = 0; i < raw_test.size(); ++i ) {

		if( i > 0 && no_space(raw_test[i], raw_test[i-1]) ) {
			std::string tt1 = " ";
			tt1 += raw_test[i];
			processed += tt1;
		} else {
			std::string tt2 = "";
			tt2 += raw_test[i];
			processed += tt2;
		}
	}

	// convert lowercase
	for(auto& c : processed) {
	   c = std::tolower(c);
	}

    return processed;
}


std::tuple<std::vector<std::vector<std::string>>, std::vector<std::vector<std::string>>> tokenize_nmt(std::string processed,
																										size_t num_examples) {
    // Tokenize the English-French dataset.
    std::vector<std::vector<std::string>> source, target;
    std::stringstream st(processed);
	std::string line;
	size_t i = 0;
	while(std::getline(st, line, '\n')) {

	     size_t pos = line.find("\t");
	     std::string part1 = line.substr(0, pos);
	     std::string part2 = line.substr(pos+1, line.size());

	     std::stringstream ss(part1);
	     std::string token;
	     std::vector<std::string> stk;
	     while(ss >> token) {
	    	 // strip space
	    	 token = std::regex_replace(token, std::regex("^\\s+"), std::string(""));
	    	 token = std::regex_replace(token, std::regex("\\s+$"), std::string(""));
	    	 stk.push_back(token);
	     }
	     source.push_back(stk);

	     std::stringstream ss2(part2);
	     std::vector<std::string> stk2;
	     while(ss2 >> token) {
	    	 // strip space
	    	 token = std::regex_replace(token, std::regex("^\\s+"), std::string(""));
	    	 token = std::regex_replace(token, std::regex("\\s+$"), std::string(""));
	    	 stk2.push_back(token);
	     }
	     target.push_back(stk2);
	     i++;
	     if( i % 10000 == 0 ) std::cout << "complete: " << i << std::endl;
	     if( num_examples > 0 && i > num_examples ) break;
	}
    return {source, target};
}

void show_list_len_pair_hist(std::vector<std::vector<std::string>> source,
							 std::vector<std::vector<std::string>> target) {
	// show_list_len_pair_hist
	std::vector<float> xs(60);
	std::vector<float> xt(60);
	std::vector<float> ys(60);
	std::vector<float> yt(60);
	for( int i = 0; i < 60; i++ ) {
		xs[i] = i*1.0 - 0.5;
		xt[i] = i*1.0 + 0.5;
		ys[i] = 0;
		yt[i] = 0;
	}

	for( size_t i = 0; i < source.size(); i++ ) {
		ys[source[i].size()] += 1;
		yt[target[i].size()] += 1;
	}

	plt::figure_size(600, 450);
	plt::bar(xs, ys, "blue", "-", 0.25, {{"label", "source"}});
	plt::bar(xt, yt, "orange", "-", 0.25, {{"label", "target"}});
	plt::xlabel("# tokens per sequence");
	plt::ylabel("Count");
	plt::legend();
	plt::show();
}

std::vector<std::pair<std::string, int64_t>> count_corpus2( std::vector<std::string> tokens ) {
	std::vector<std::pair<std::string, int64_t>> _token_freqs;
	std::map<std::string, int64_t> cntTks;


    for (auto tk : tokens) {

        if (cntTks.find(tk) == cntTks.end()) {  // if key is NOT present already
        	cntTks[tk] = 1; 					// initialize the key with value 1
        } else {
        	cntTks[tk]++; 						// key is already present, increment the value by 1
        }
    }

    // copy key-value pairs from the map to the vector
    std::copy(cntTks.begin(), cntTks.end(), std::back_inserter<std::vector<std::pair<std::string, int64_t>>>(_token_freqs));

    // sort the vector by increasing the order of its pair's second value
    // if the second value is equal, order by the pair's first value
    std::sort(_token_freqs.begin(), _token_freqs.end(),
                      [](const std::pair<std::string, int64_t> &l, const std::pair<std::string, int64_t> &r) {
                          if (l.second != r.second) {
                              return l.second > r.second;
                          }
                          return l.first > r.first;
                      });

    return _token_freqs;
}

template<typename T>
std::vector<T> truncate_pad(std::vector<T> line, size_t num_steps, T padding_token) {
    //Truncate or pad sequences."""
    if( line.size() > num_steps ) {
    	std::vector<T> tokens(&line[0], &line[num_steps]);
        return tokens;  // Truncate
    } else {
    	int num_pad = num_steps - line.size();
    	for( int i = 0; i < num_pad; i++ )
    		line.push_back(padding_token);
    	return line;
    }
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(7);
	std::string filename = "./data/fra-eng/fra.txt";

	std::string raw_test = read_data_nmt(filename);

	std::cout << raw_test.size() << std::endl;
	std::cout << raw_test.substr(0, 75) << std::endl;

	std::string processed = preprocess_nmt( raw_test );

	std::cout << processed.size() << std::endl;
	std::cout << processed.substr(0, 75) << std::endl;

	std::vector<std::vector<std::string>> source, target;
	std::tie(source, target) = tokenize_nmt(processed, 10000); // set 0 to use all samples

	std::cout << source.size() << std::endl;
	std::cout << target.size() << std::endl;

//	show_list_len_pair_hist(source, target);

	std::vector<std::string> tokens;
	for(std::vector<std::string>& tk : source ) {
		for(std::string& t : tk) {
			tokens.push_back(t);
		}
	}

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> rv({"<pad>", "<bos>", "<eos>"});
	auto vocab = Vocab(counter, 2.0, rv);
	std::cout << vocab.length() << "\n";
	std::cout << vocab.to_tokens({47}) << "\n";

	for(std::string& t : source[0])
		std::cout << t << " " << vocab[t] << " ";
	std::cout << "\n";

	auto idx = vocab[source[0]];
	std::cout << idx.size() << "\n";
	for(auto& t : idx)
		std::cout << t << " ";
	std::cout << "\n";

	auto rlt = truncate_pad(vocab[source[0]], 10, vocab["<pad>"]);
	for(auto& t : rlt)
		std::cout << t << " ";
	std::cout << "\n";



	std::cout << "Done!\n";
	return 0;
}



