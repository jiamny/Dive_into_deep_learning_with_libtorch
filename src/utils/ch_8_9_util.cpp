#include "ch_8_9_util.h"


std::vector<std::string> read_time_machine( std::string filename ) {
	std::vector<std::string> lines;
	std::ifstream myReadFile;
	std::regex reg("[^A-Za-z]+");
	myReadFile.open(filename.c_str());

	std::string output;
	std::locale loc;

	if(myReadFile.is_open()) {
		 while( ! myReadFile.eof() ) {
			 //myReadFile >> output;
			 getline(myReadFile, output); // Saves the line in STRING.
			 output = std::regex_replace(output, reg, " ");
			 // strip space
			 //output.erase(std::remove_if(output.begin(), output.end(), isspace), output.end());
			 output = std::regex_replace(output, std::regex("^\\s+"), std::string(""));
			 output = std::regex_replace(output, std::regex("\\s+$"), std::string(""));
			 // convert lowercase
			 for(auto& c : output)
				 c = std::tolower(c);
			 //std::cout << output << std::endl;
			 lines.push_back(std::move(output));
		 }
	}
	myReadFile.close();
	return lines;
}

std::vector<std::string> tokenize(const std::vector<std::string> lines, const std::string token, bool max_cut) {
	std::string item;
	std::vector<std::string> elems;
	int max_tokens = 10000;
	int cnt = 0;
	if( token == "word" ) {
		for( int i = 0; i < lines.size(); i++ ) {
			std::stringstream ss(lines[i]);
			char delim = ' ';
			while (std::getline(ss, item, delim)) {
				// trim space
				item = std::regex_replace(item, std::regex("^\\s+"), std::string(""));
				item = std::regex_replace(item, std::regex("\\s+$"), std::string(""));
			    //elems.push_back(item);
				if( max_cut ) {
					if( cnt < max_tokens )
						elems.push_back(std::move(item));
				} else {
					elems.push_back(std::move(item)); // if C++11 (based on comment from @mchiasson)
				}
				cnt++;
			}
		}
	} else if( token == "char" ) {
		for( int i = 0; i < lines.size(); i++ ) {
			std::vector<char> v(lines[i].begin(), lines[i].end());
			for(int i = 0; i < v.size(); i++ ) {
				std::string tc(1, v[i]);

				if( max_cut ) {
					if( cnt < max_tokens ) elems.push_back(tc);
				} else {
					elems.push_back(tc);
				}
			}
		}
	} else {
		std::cout << "ERROR: unknown token type: " + token << std::endl;
	}

    return elems;
}


std::vector<std::pair<std::string, int64_t>> count_corpus( std::vector<std::string> tokens ) {
	std::vector<std::pair<std::string, int64_t>> _token_freqs;
	std::map<std::string, int64_t> cntTks;

    //Count token frequencies
    // Here `tokens` is a 1D list or 2D list
	/*
    if len(tokens) == 0 or isinstance(tokens[0], list):
        // Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    */
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


Vocab::Vocab(std::vector<std::pair<std::string, int64_t>> corpus, float min_freq,
		std::vector<std::string> reserved_tokens) {

	_token_freqs = corpus;
    idx_to_token.clear();
    token_to_idx.clear();

    // create an empty map of pairs
    if( ! token_to_idx.empty() ) token_to_idx.clear();
    if( ! token_to_idx.empty() ) token_to_idx.clear();

    token_to_idx.insert(std::make_pair("<unk>", 0));
    idx_to_token.insert(std::make_pair(0, "<unk>"));

    int64_t idx = 1;
    if( ! reserved_tokens.empty()) {
    	for( auto& rt : reserved_tokens ) {
    		token_to_idx.insert(std::make_pair(rt, idx));
    		idx_to_token.insert(std::make_pair(idx, rt));
    		idx++;
    	}
    }

    for(const auto it : _token_freqs ) {
    	// not find
    	if( token_to_idx.find(it.first) == token_to_idx.end() ) {
    		// not less than min freq
    		if( it.second >= min_freq ) {
    			token_to_idx.insert(std::make_pair(it.first, idx));
    			idx_to_token.insert(std::make_pair(idx, it.first));
    			idx++;
    		}
    	}
    }

    // sorted by value
    order_token_to_idx = std::set<std::pair<std::string, int64_t>, comp>(token_to_idx.begin(), token_to_idx.end());
}


int64_t Vocab::length(void) {
    return idx_to_token.size();
}

int64_t Vocab::unk(void){  // Index for the unknown token
    return 0;
}

std::vector<std::string> Vocab::to_tokens( std::vector<int64_t> indices ) {
    std::vector<std::string> values;
    auto it = idx_to_token.begin();

    for( int64_t i = 0; i < indices.size(); i++ ) {
    	std::advance(it, indices[i]);
    	values.push_back( it->second );
    }

    return values;
}

std::vector<std::pair<std::string, int64_t>> Vocab::token_freqs(void) {
    return _token_freqs;
}


// Overload the + operator
int64_t Vocab::operator [] (const std::string s) {

    if( token_to_idx.find(s) == token_to_idx.end() ) {
    	return 0;
    } else {
    	auto it = token_to_idx.find(s);
    	return it->second;
    }
}

// Overload the [] operator
std::vector<int64_t> Vocab::operator [] (const std::vector<std::string> ss ) {
	std::vector<int64_t> idx;
	if( ! ss.empty() ) {
		for( auto& s : ss ) {
			if( token_to_idx.find(s) == token_to_idx.end() ) {
				idx.push_back(0);
			} else {
		    	auto it = token_to_idx.find(s);
		    	idx.push_back(it->second);
			}
		}
	} else {
		idx.push_back(0);
	}
	return idx;
}

// ----------------------------------------------------------
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

std::tuple<torch::Tensor, torch::Tensor> build_array_nmt(std::vector<std::vector<std::string>> lines,
														 Vocab vocab, int num_steps) {
    //将机器翻译的文本序列转换成小批量
	std::vector<int64_t> vec;
	int row = 0;
	for(auto& l : lines) {
		std::vector<int64_t> a = vocab[l];
		a.push_back(vocab["<eos>"]);
		auto c = truncate_pad( a, num_steps, vocab["<pad>"]);
		for(auto i : c)
			vec.push_back(i);
		row++;
	}
	auto options = torch::TensorOptions().dtype(torch::kLong);
	torch::Tensor array = torch::from_blob(vec.data(), {row, num_steps}, options);
	array = array.clone().to(torch::kLong);
	torch::Tensor valid_len = (array != vocab["<pad>"]).to(torch::kLong).sum(1);
    return {array, valid_len};
}

Nmtdataset::Nmtdataset(std::pair<torch::Tensor, torch::Tensor> data_arrays) {
    features_ = std::move(data_arrays.first);
    labels_   = std::move(data_arrays.second);
}

torch::data::Example<> Nmtdataset::get(size_t index) {
    return {features_[index], labels_[index]};
}

torch::optional<size_t> Nmtdataset::size() const {
    return features_.size(0);
}

const torch::Tensor& Nmtdataset::features() const {
    return features_;
}

const torch::Tensor& Nmtdataset::labels() const {
    return labels_;
}


std::tuple<std::pair<torch::Tensor, torch::Tensor>, Vocab, Vocab> load_data_nmt( std::string filename,
		int num_steps, int num_examples=600) {
    //返回翻译数据集的迭代器和词表
	std::string raw_test = read_data_nmt(filename);
	std::string processed = preprocess_nmt( raw_test );

	std::vector<std::vector<std::string>> source, target;
	std::tie(source, target) = tokenize_nmt(processed, num_examples); // set 0 to use all samples

	std::vector<std::string> tokens;
	for(std::vector<std::string>& tk : source ) {
		for(std::string& t : tk) {
			tokens.push_back(t);
		}
	}

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );
	std::vector<std::string> rv({"<pad>", "<bos>", "<eos>"});
	auto src_vocab = Vocab(counter, 2.0, rv);

	// transform text sequences into minibatches for training.
	torch::Tensor src_array, src_valid_len;
	std::tie(src_array, src_valid_len) = build_array_nmt(source, src_vocab, num_steps);

	std::vector<std::string> tgt_tokens;
	for(std::vector<std::string>& tk : target ) {
		for(std::string& t : tk) {
			tgt_tokens.push_back(t);
		}
	}
	std::vector<std::pair<std::string, int64_t>> tgt_counter = count_corpus( tgt_tokens );
	auto tgt_vocab = Vocab(tgt_counter, 2.0, rv);

	torch::Tensor tgt_array, tgt_valid_len;
	std::tie(tgt_array, tgt_valid_len) = build_array_nmt(target, tgt_vocab, num_steps);

	// merge tensors
	torch::Tensor features = torch::cat({src_array, tgt_array}, -1);

	torch::Tensor labels = (torch::cat({src_valid_len, tgt_valid_len}, -1).reshape({-1, src_valid_len.size(0)})).transpose(1, 0);

	std::pair<torch::Tensor, torch::Tensor> data_arrays = {features, labels};

    return {data_arrays, src_vocab, tgt_vocab};
}



