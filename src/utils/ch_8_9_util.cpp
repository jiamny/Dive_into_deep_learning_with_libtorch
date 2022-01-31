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



