/*
 * SubwordEmbedding.cpp
 *
 */
#include <unistd.h>
#include <iomanip>
#include <algorithm>
#include <cmath>
#include <torch/utils.h>
#include <torch/torch.h>
#include <string>

#include "../utils/ch_14_util.h"

std::string join(std::vector<std::string> tks, std::string sep) {

	std::string mg = "";
	for(int i = 0; i < tks.size(); i++ ) {
		if( i == (tks.size()-1))
			mg += tks[i];
		else {
			mg += (tks[i] + sep);
		}
	}
	return mg;
}

typedef struct freq_pair {
	std::string fst = "";
	std::string snd = "";
	int64_t freq = 0;
	freq_pair(void) {}

	freq_pair(std::string s1, std::string s2) {
		fst = s1;
		snd = s2;
	}

	freq_pair(std::string s1, std::string s2, int64_t fq) {
		fst = s1;
		snd = s2;
		freq = fq;
	}

} freq_pair;

bool cmp(freq_pair a, freq_pair b) {
	return a.freq > b.freq;
}

freq_pair maxpair( std::vector<freq_pair> mp ) {
	/*
	std::vector< std::pair<std::string, int64_t> > vec;

	for(std::map<std::string, int64_t>::iterator it = mp.begin();
			it != mp.end(); it++) {
		vec.push_back(std::pair<std::string, int64_t>(it->first, it->second));
	}

	std::sort(vec.begin(), vec.end(), cmp);
	vec[0].first;
	*/
	int64_t max_freq = 0;
	freq_pair max_pair;

	for(int i = 0; i < mp.size(); i++) {
		int64_t freq = mp[i].freq;
		if( freq > max_freq ) {
			max_pair = mp[i];
			max_freq = freq;
		}
	}
	return max_pair;
}

int findVectorElement(std::vector<freq_pair> mp, freq_pair target) {
	int idx = -1;
	for(int i = 0; i < mp.size(); i++) {
		if( mp[i].fst == target.fst && mp[i].snd == target.snd ) {
			idx = i;
			break;
		}
	}
	return idx;
}

std::vector<std::string> strsplit(const std::string& str, char delim) {
    std::vector<std::string> elems;
    // empty char
    if( delim == 'e' ) {
    	const int length = str.length();
    	// declaring character array (+1 for null terminator)
    	char* char_array = new char[length + 1];
    	// copying the contents of the
    	// string to char array
    	strcpy(char_array, str.c_str());
    	for(int i = 0; i < length; i++ ) {
    		std::string s = "";
    		s.push_back(char_array[i]);
    		elems.push_back(s);
    	}
    } else {
    	std::stringstream ss(str);
    	std::string item;
    	while (std::getline(ss, item, delim)) {
    		if (!item.empty()) {
    			elems.push_back(item);
    		}
    	}
    }
    return elems;
}


freq_pair get_max_freq_pair(std::map<std::string, int64_t> token_freqs) {

	std::vector<freq_pair> pairs;

    for( auto& [token, freq] : token_freqs ) {

    	std::vector<std::string> symbols = strsplit(token, ' ');
    	//std::cout << '\n';
    	//std::for_each(symbols.begin(), symbols.end(),
    	//	                [](const std::string &p) {std::cout << p << " ";});
        //std::cout << '\n';

        for( int i = 0; i < (symbols.size() - 1); i++ ) {
            // Key of `pairs` is a tuple of two consecutive symbols
        	freq_pair pr = freq_pair(symbols[i], symbols[i + 1]);

        	if( pairs.size() < 1 ) {
        		pr.freq = freq;
        		pairs.push_back(pr);
        	} else {
        		int idx = findVectorElement(pairs, pr);
        		if( idx < 0 ) {
	        		pr.freq = freq;
	        		pairs.push_back(pr);
	        	} else {
	        		pairs[idx].freq += freq;
	        	}
        	}

        }
    }
    std::cout << "pairs: ";
	std::for_each(pairs.begin(), pairs.end(),
	                [](const freq_pair& p) {
	                    std::cout << "{" << p.fst << ", " << p.snd  << ", " << p.freq << "} ";});
	std::cout << '\n';
    return maxpair( pairs );  // Key of `pairs` with the max value
}

std::map<std::string, int64_t> merge_symbols(freq_pair max_freq_pair, std::map<std::string, int64_t> token_freqs,
		std::vector<std::string>& symbols) {
	std::vector<std::string> prs = {max_freq_pair.fst, max_freq_pair.snd};
	symbols.push_back( join(prs, "") );
	std::map<std::string, int64_t> new_token_freqs;

	for( auto& [token, freq] : token_freqs ) {
		 std::cout << "token: \'" << token << "\' max_freq_pair: [\'" << max_freq_pair.fst << "\', \'"
				 << max_freq_pair.snd << "\']\n";

		std::string o_tk = join(prs, " ");
		std::string n_tk = join(prs, "");
		std::string new_token = token;
		int tkL = new_token.length();

		if( token.find(o_tk) != std::string::npos ) {
			int lnd = token.find(o_tk);
			int rst = lnd + o_tk.length();
			std::string lft = token.substr(0, lnd);

			std::string rgt = "";
			if( lnd < tkL )
				rgt = token.substr(rst);

			if( lft.empty() ) {
				if( ! rgt.empty() )
					new_token = n_tk + " " + strip(rgt);
				else
					new_token = n_tk;
			} else {
				if( ! rgt.empty() )
					new_token = strip(lft) + " " + n_tk + " " + strip(rgt);
				else
					new_token = strip(lft) + " " + n_tk;
			}

			std::cout << "o_tk: \'" << o_tk <<
					"\' new_token: \'" << new_token << "\' token: \'" << token << "\'\n";
		}

		new_token_freqs[new_token] = token_freqs[token];
	}
	return new_token_freqs;
}


std::vector<std::string> segment_BPE(std::vector<std::string> tokens, std::vector<std::string> symbols) {
	std::vector<std::string> outputs;
    for(auto& token : tokens ) {
        int start = 0, end = token.length();
        std::vector<std::string> cur_output;
        // Segment token with the longest possible subwords from symbols
        while( start < token.length() && start < end ) {
        	int len = end - start;
        	std::string substr = token.substr(start, len);
        	if( std::find(symbols.begin(), symbols.end(), substr) != symbols.end() ) {
                cur_output.push_back(substr);
                start = end;
                end = token.length();
            } else {
                end -= 1;
            }
        }

        if( start < token.length() )
            cur_output.push_back("[UNK]");

        outputs.push_back(join(cur_output, " "));
    }
    return outputs;
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << "\n";

	std::vector<std::string> symbols = {"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
	           "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
	           "_", "[UNK]"};

	std::map<std::string, int64_t> raw_token_freqs = {
		{"fast_", 4},
		{"faster_", 3},
		{"tall_", 5},
		{"taller_", 4}};

	std::map<std::string, int64_t> token_freqs;

	for( auto& [tk, freq] : raw_token_freqs ) {
		std::vector<std::string> ss = strsplit(tk, 'e');
		std::string mg = join(ss, " ");
		token_freqs[mg] = raw_token_freqs[tk];
	}

	std::for_each(token_freqs.begin(), token_freqs.end(),
	                [](const std::pair<std::string, int64_t> &p) {
	                    std::cout << "{" << p.first << ": " << p.second << "} ";});
	std::cout << '\n';

	std::vector<freq_pair> ele = {
			{freq_pair("a", "x", 5)},
			{freq_pair("b", "y", 6)},
			{freq_pair("c", "z", 4)}};

	freq_pair mxp = maxpair(ele);
	std::cout << "{" << mxp.fst << ", " << mxp.snd << ", " << mxp.freq << "}\n";

	int num_merges = 10;
	for(int i = 0;  i < num_merges; i++) {
	    std::cout << "---------------------------------------------> " << (i+1) << '\n';
	    std::cout << "token_freq: ";
		std::for_each(token_freqs.begin(), token_freqs.end(),
		                [](const std::pair<std::string, int64_t> &p) {
		                    std::cout << "{" << p.first << ": " << p.second << "} ";});
		std::cout << '\n';

	    freq_pair max_freq_pair = get_max_freq_pair(token_freqs);
	    std::cout << "max_freq_pair: " << max_freq_pair.fst << " " << max_freq_pair.snd << '\n';
	    token_freqs = merge_symbols(max_freq_pair, token_freqs, symbols);
	}


	std::cout << "token_freqs.keys: [";
    std::for_each(token_freqs.begin(), token_freqs.end(),
    	                [](const std::pair<std::string, int64_t> &p) {
    	                    std::cout << "\'" <<p.first << "\' ";});
    std::cout << '\n';

    std::cout << "symbols: [";
    std::for_each(symbols.begin(), symbols.end(),
    		[](const std::string &p) {std::cout << "\'" << p << "\' ";});
    std::cout << "]\n";


    std::vector<std::string> tokens = {"tallest_", "fatter_"};
    std::vector<std::string> ot = segment_BPE(tokens, symbols);

    std::cout << "\nsegment_BPE: [";
    std::for_each(ot.begin(), ot.end(),
        		[](const std::string &p) {std::cout << "\'" << p << "\' ";});
    std::cout << "]\n";

	std::cout << "\nDone!\n";

	return 0;
}
