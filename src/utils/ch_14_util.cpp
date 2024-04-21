#include "ch_14_util.h"

std::string strip( const std::string& s ) {
	const std::string WHITESPACE = " \n\r\t\f\v";

	size_t start = s.find_first_not_of(WHITESPACE);
	std::string ls = (start == std::string::npos) ? "" : s.substr(start);

	size_t end = ls.find_last_not_of(WHITESPACE);
	return (end == std::string::npos) ? "" : ls.substr(0, end + 1);
}

std::vector<std::vector<std::vector<std::string>>> _read_wiki(const std::string data_dir, size_t num_read) {
	std::vector<std::vector<std::vector<std::string>>> data;
	std::string f = data_dir + "/wiki.train.tokens";
    std::string line;
    const std::string WHITESPACE = " ";

    std::ifstream fL(f.c_str());

    if( fL.is_open() ) {
    	size_t cnt = 0;

    	while ( std::getline(fL, line) ) {

    		line = std::regex_replace(line, std::regex("\\\n"), "");

    		line = strip(line);
    		transform(line.begin(), line.end(), line.begin(), ::tolower);

    		std::string space_delimiter = " . ";
    		std::vector<std::vector<std::string>> words{};

    		size_t pos = 0;
    		while ((pos = line.find(space_delimiter)) != std::string::npos) {
    			std::string s = line.substr(0, pos);
    			if( s.length() > 1 ) {
    				std::vector<std::string> tks{};
    				size_t ps = 0;

    				while ((ps = s.find(WHITESPACE)) != std::string::npos) {
    					std::string tk = s.substr(0, ps);
    					// trim space
    					tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    					tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    					if( tk.length() > 1 )
    						tks.push_back(tk);
    					s.erase(0, ps + WHITESPACE.length());
    				}

    				std::string tk = s.substr(0, ps);
    				// trim space
    				tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    				tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    				if( tk.length() > 1 )
    				    tks.push_back(tk);

    				if(tks.size() > 0 ) words.push_back(tks);
    			}

    			line.erase(0, pos + space_delimiter.length());
    		}
    		std::string s = line.substr(0, pos);
    		if( s.length() > 1 ) {
    			std::vector<std::string> tks{};
    			size_t ps = 0;

    			while ((ps = s.find(WHITESPACE)) != std::string::npos) {
    			    std::string tk = s.substr(0, ps);
    			    // trim space
    			    tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    			    tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    			    if( tk.length() > 1 )
    			    	tks.push_back(tk);
    			    s.erase(0, ps + WHITESPACE.length());
    			}

    			std::string tk = s.substr(0, ps);
    			// trim space
    			tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    			tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    			if( tk.length() > 1 )
    			    tks.push_back(tk);

    			if(tks.size() > 0 ) words.push_back(tks);
    		}

    		if( words.size() >= 2 ) {
    			data.push_back(words);
    		}

    		cnt++;
    		if( num_read > 0 && cnt > num_read)
    			break;
    	}
    }

    fL.close();

    std::random_shuffle(data.begin(), data.end());
    return data;
}

std::pair<std::vector<std::string>, std::vector<int64_t>> get_tokens_and_segments(std::vector<std::string> tokens_a,
																				std::vector<std::string> tokens_b) {
    // Get tokens of the BERT input sequence and their segment IDs.
	// 获取输入序列的词元及其片段索引;
	size_t org_tk_size = tokens_a.size();
	auto iter = tokens_a.insert(std::begin(tokens_a), "<cls>");
	tokens_a.push_back("<sep>");

    // 0 and 1 are marking segment A and B, respectively
    //segments = [0] * (len(tokens_a) + 2)
    std::vector<int64_t> segments;
    for( int i = 0; i < (org_tk_size + 2); i++ )
    	segments.push_back(0);

    if( ! tokens_b.empty() ) {
        //tokens += tokens_b + ['<sep>']
    	for(auto& s :tokens_b)
    	    tokens_a.push_back(s);
    	tokens_a.push_back("<sep>");

        //segments += [1] * (len(tokens_b) + 1)
    	for( int i = 0; i < (tokens_b.size() + 1); i++ )
    	    	segments.push_back(1);
    }
    return std::make_pair(tokens_a, segments);
}

std::tuple<std::vector<std::string>, std::vector<std::string>, bool> _get_next_sentence(std::vector<std::string> sentence,
							std::vector<std::string> next_sentence, std::vector<std::vector<std::vector<std::string>>> paragraphs) {
	std::random_device rd{};
	// Use Mersenne twister engine to generate pseudo-random numbers.
	std::mt19937 engine{rd()};

	std::uniform_real_distribution<double> dist{0.0, 1.0};

	bool is_next = false;

    if( dist(engine) < 0.5 )
        is_next = true;
    else {
        // `paragraphs` is a list of lists of lists
        //next_sentence = random.choice(random.choice(paragraphs))
    	std::srand(std::time(0)); // use current time as seed for random generator
    	int p_pos = std::rand() % paragraphs.size();
    	std::vector<std::vector<std::string>> random_val = paragraphs[p_pos];

    	int s_pos = std::rand() % random_val.size();
    	next_sentence = random_val[s_pos];

        is_next = false;
    }
    return std::make_tuple(sentence, next_sentence, is_next);
}


std::vector<std::tuple<std::vector<std::string>, std::vector<int64_t>, bool>> _get_nsp_data_from_paragraph(
		std::vector<std::vector<std::string>> paragraph, std::vector<std::vector<std::vector<std::string>>> paragraphs,
		Vocab vocab, size_t max_len) {

	std::vector<std::tuple<std::vector<std::string>, std::vector<int64_t>, bool>> nsp_data_from_paragraph;

    std::vector<std::string> tokens_a, tokens_b;
    bool is_next;
    for(int i = 0; i < (paragraph.size() - 1); i++) {
        std::tie(tokens_a, tokens_b, is_next) = _get_next_sentence(
            paragraph[i], paragraph[i + 1], paragraphs);

        // Consider 1 '<cls>' token and 2 '<sep>' tokens
        if( (tokens_a.size() + tokens_b.size() + 3) > max_len )
            continue;

        auto rlt = get_tokens_and_segments(tokens_a, tokens_b);
        std::vector<std::string> tokens = rlt.first;
        std::vector<int64_t> segments = rlt.second;
        nsp_data_from_paragraph.push_back(std::make_tuple(tokens, segments, is_next));
    }
    return nsp_data_from_paragraph;
}

std::pair<std::vector<std::string>, std::map<int64_t, std::string> > _replace_mlm_tokens(std::vector<std::string> tokens,
		std::vector<int64_t> candidate_pred_positions, int64_t num_mlm_preds, Vocab vocab) {

	std::random_device rd{};
	std::mt19937 engine{rd()};
	std::uniform_real_distribution<double> dist{0.0, 1.0};

    // Make a new copy of tokens for the input of a masked language model,
    // where the input may contain replaced '<mask>' or random tokens
	std::vector<std::string> mlm_input_tokens;
	for(auto& c : tokens)
		mlm_input_tokens.push_back(c);

    //mlm_input_tokens = [token for token in tokens]
    //pred_positions_and_labels = []
	std::map<int64_t, std::string, std::less<int64_t>> pred_positions_and_labels;

    // Shuffle for getting 15% random tokens for prediction in the masked
    // language modeling task
    // random.shuffle(candidate_pred_positions)
	std::random_shuffle(candidate_pred_positions.begin(), candidate_pred_positions.end());

    for( auto& mlm_pred_position : candidate_pred_positions ) {
        if( pred_positions_and_labels.size() >= num_mlm_preds)
            break;
        std::string masked_token = "";
        // 80% of the time: replace the word with the '<mask>' token
        if( dist(engine) < 0.8 )
            masked_token = "<mask>";
        else {
            // 10% of the time: keep the word unchanged
            if( dist(engine) < 0.5 )
                masked_token = tokens[mlm_pred_position];
            // 10% of the time: replace the word with a random word
            else {
            	std::srand(std::time(0));
            	int64_t p_pos = std::rand() % vocab.length();
                masked_token = vocab.idx_to_token[p_pos];
            }
        }
        mlm_input_tokens[mlm_pred_position] = masked_token;
        //pred_positions_and_labels.append(
        //    (mlm_pred_position, tokens[mlm_pred_position]))
        pred_positions_and_labels.insert(std::make_pair(mlm_pred_position, tokens[mlm_pred_position]));
    }
    return std::make_pair(mlm_input_tokens, pred_positions_and_labels);

}

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> _get_mlm_data_from_tokens(
		std::vector<std::string> tokens, Vocab vocab) {

	std::random_device rd{};
	std::mt19937 engine{rd()};
	std::uniform_real_distribution<double> dist{0.0, 1.0};

    std::vector<int64_t> candidate_pred_positions;
    // `tokens` is a list of strings
    for(int i = 0;  i < tokens.size(); i++) {
    	auto token = tokens[i];
        // Special tokens are not predicted in the masked language modeling task
        if( token == "<cls>" || token  == "<sep>" )
            continue;
        candidate_pred_positions.push_back(i);
    }
    // 15% of random tokens are predicted in the masked language modeling task
    int num_mlm_preds = std::max(1, static_cast<int>(std::round(tokens.size() * 0.15)));

    auto gdt = _replace_mlm_tokens( tokens, candidate_pred_positions, num_mlm_preds, vocab );
    auto mlm_input_tokens = gdt.first;
    auto pred_positions_and_labels = gdt.second;

    //pred_positions_and_labels = sorted(pred_positions_and_labels,
    //                                   key=lambda x: x[0])

    //pred_positions = [v[0] for v in pred_positions_and_labels]
    //mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    std::vector<int64_t> pred_positions;
    std::vector<std::string> mlm_pred_labels;

    for(auto& v : pred_positions_and_labels) {
    	pred_positions.push_back(v.first);
    	mlm_pred_labels.push_back(v.second);
    }

    return std::make_tuple(vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor>
_pad_bert_inputs(std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
							 std::vector<int64_t>, std::vector<int64_t>, bool>> examples,
		size_t max_len, Vocab vocab) {
	size_t max_num_mlm_preds = static_cast<size_t>(round(max_len * 0.15));
    //all_token_ids, all_segments, valid_lens,  = [], [], []
    //all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    std::vector<torch::Tensor> all_token_ids, all_segments, all_pred_positions, all_mlm_weights, all_mlm_labels;
    std::vector<torch::Tensor> valid_lens, nsp_labels; // = []
    //for (token_ids, pred_positions, mlm_pred_label_ids, segments,
    //     is_next) in examples:
    std::vector<int64_t> token_ids, pred_positions, mlm_pred_label_ids, segments, idxs;
    bool is_next;
    int idx = 0;
    for( auto& exp : examples ) {
        token_ids = std::get<0>(exp);
        pred_positions = std::get<1>(exp);
        mlm_pred_label_ids = std::get<2>(exp);
        segments = std::get<3>(exp);
        is_next = std::get<4>(exp);

        size_t org_tk_ids = token_ids.size();
        if( org_tk_ids < max_len ) {
        	for(size_t i = 0; i < (max_len - org_tk_ids); i++ )
        		token_ids.push_back(vocab["<pad"]);
        }
//        std::cout << "org_tk_ids.size: " << org_tk_ids << " ; token_ids.size: " << token_ids.size() << '\n';

        torch::Tensor t = torch::from_blob(token_ids.data(), {1, static_cast<int>(token_ids.size())}, dtype(torch::kLong)).clone();
        all_token_ids.push_back(t.clone());

        size_t org_segments = segments.size();
        if(org_segments  < max_len ) {
        	for(size_t i = 0; i < (max_len - org_segments); i++ )
        		segments.push_back(0);
        }
//        std::cout << "segments.size: " << segments.size() << '\n';

        t = torch::from_blob(segments.data(), {1, static_cast<int>(segments.size())}, dtype(torch::kLong)).clone();
        all_segments.push_back(t.clone());

        valid_lens.push_back(torch::full({1}, static_cast<int>(org_tk_ids)).to(torch::kFloat32).clone());

        size_t org_pred_positions = pred_positions.size();
        if( org_pred_positions < max_num_mlm_preds) {
        	for(size_t i = 0; i < (max_num_mlm_preds - org_pred_positions); i++ )
        		pred_positions.push_back(0);
        }
//        std::cout << "pred_positions.sizet: " << pred_positions.size() << '\n';

        t = torch::from_blob(pred_positions.data(), {1, static_cast<int>(pred_positions.size())}, dtype(torch::kLong)).clone();
        all_pred_positions.push_back(t.clone());

        // Predictions of padded tokens will be filtered out in the loss via
        // multiplication of 0 weights
        std::vector<float> vt;
        for(int i = 0; i < mlm_pred_label_ids.size(); i++)
        	vt.push_back(1.0f);

        for(int i = 0; i < (max_num_mlm_preds - org_pred_positions); i++)
            vt.push_back(0.0f);
//        std::cout << "all_mlm_weights.size: " << vt.size() << '\n';

        t = torch::from_blob(vt.data(), {1, static_cast<int>(vt.size())}, dtype(torch::kFloat32)).clone();
        all_mlm_weights.push_back(t.clone());

        size_t org_mlm_pred_label_ids = mlm_pred_label_ids.size();
        if( org_mlm_pred_label_ids < max_num_mlm_preds ) {
        	for( int i = 0; i < (max_num_mlm_preds - org_mlm_pred_label_ids); i++ )
        		mlm_pred_label_ids.push_back(0);
        }
//        std::cout << "mlm_pred_label_ids.size: " << mlm_pred_label_ids.size() << '\n';

        t = torch::from_blob(mlm_pred_label_ids.data(), {1, static_cast<int>(mlm_pred_label_ids.size())}, dtype(torch::kLong)).clone();
        all_mlm_labels.push_back(t.clone());

//        std::cout << "is_next: " << is_next << '\n';
		if( is_next )
			nsp_labels.push_back(torch::full({1}, 1).to(torch::kLong).clone());
		else
			nsp_labels.push_back(torch::full({1}, 0).to(torch::kLong).clone());

		idxs.push_back(idx);
		idx += 1;
    }

    torch::Tensor tidxs = torch::from_blob(idxs.data(),
    							 {static_cast<long>(idxs.size()), 1}, torch::TensorOptions(torch::kLong)).clone();
//    std::cout << "1. - tidxs.sizes(): " << tidxs.sizes() << '\n';

    torch::Tensor d1=torch::concat(all_token_ids, 0).to(torch::kLong);
    torch::Tensor d2=torch::concat(all_segments, 0).to(torch::kLong);
    torch::Tensor d3=torch::concat(valid_lens, 0).to(torch::kFloat32);
    torch::Tensor d4=torch::concat(all_pred_positions, 0).to(torch::kLong);
    torch::Tensor d5=torch::concat(all_mlm_weights, 0).to(torch::kFloat32);
    torch::Tensor d6=torch::concat(all_mlm_labels, 0).to(torch::kLong);
    torch::Tensor d7=torch::concat(nsp_labels, 0).to(torch::kLong);
//    std::cout << "concat done!\n";
    return std::make_tuple(d1, d2, d3, d4, d5, d6, d7, tidxs);
}

torch::Tensor  transpose_output(torch::Tensor X, int64_t num_heads) {
    //逆转 `transpose_qkv` 函数的操作。
    X = X.reshape({-1, num_heads, X.size(1), X.size(2)});
    X = X.permute({0, 2, 1, 3});//X.transpose(0, 2, 1, 3);
    return X.reshape({X.size(0), X.size(1), -1});
}

torch::Tensor transpose_qkv(torch::Tensor X, int64_t num_heads) {
    //为了多注意力头的并行计算而变换形状。
    X = X.reshape({X.size(0), X.size(1), num_heads, -1});
    X = X.permute({0, 2, 1, 3}); //transpose(0, 2, 1, 3)
    return X.reshape({-1, X.size(2), X.size(3)});
}

std::vector<std::vector<std::string>> read_ptb(const std::string data_dir, size_t num_read) {
	std::vector<std::vector<std::string>> data;
	std::string f = data_dir + "/ptb.train.txt";
    std::string line;
    const std::string WHITESPACE = " ";

    std::ifstream fL(f.c_str());

    if( fL.is_open() ) {
    	size_t cnt = 0;

    	while ( std::getline(fL, line) ) {

    		line = std::regex_replace(line, std::regex("\\\n"), "");

    		line = strip(line);
    		std::vector<std::string> words;

    		size_t ps = 0;
    		while ((ps = line.find(WHITESPACE)) != std::string::npos) {
    			std::string tk = line.substr(0, ps);
    			tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    			tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    			if( tk.length() > 1 ) {
    			    words.push_back(tk);
    			}
    			line.erase(0, ps + WHITESPACE.length());
    		}
    		std::string tk = line.substr(0, ps);
    	    // trim space
    	    tk = std::regex_replace(tk, std::regex("^\\s+"), std::string(""));
    	    tk = std::regex_replace(tk, std::regex("\\s+$"), std::string(""));
    	    if( tk.length() > 1 )
    	    	words.push_back(tk);

    	    if( words.size() > 0 )
    	    	data.push_back(words);

        	cnt++;
        	if( num_read > 0 && cnt > num_read)
        		break;
    	}
    }

    fL.close();
    return data;
}

// Return True if `token` is kept during subsampling
bool keep(std::string token, std::map<std::string, int64_t> counter, int64_t num_tokens) {
	std::default_random_engine random(time(NULL));
	std::uniform_real_distribution<double> dis(0.0, 1.0);

    return( dis(random) <
           std::sqrt(static_cast<double>(1e-4 *1.0 / counter[token] * num_tokens)));
}

std::pair<std::vector<std::vector<std::string>>, std::map<std::string, int64_t>>
subsample(std::vector<std::vector<std::string>> sentences, Vocab vocab,
		std::vector<std::pair<std::string, int64_t>> cnt_corpus) {
    //Subsample high-frequency words.
    // Exclude unknown tokens ('<unk>')
	std::map<std::string, int64_t> counter;
	std::vector<std::vector<std::string>> slt_sentences;
	std::set<std::string> slt_corpus;

	for(int64_t r = 0; r < sentences.size(); r++) {
		std::vector<std::string> dt;
		for(auto& d : sentences[r]) {
			if( vocab[d] != vocab.unk() ) {
				dt.push_back(d);
				if( slt_corpus.find(d) != slt_corpus.end()) {
					slt_corpus.insert(d);
				}
			}
		}
		slt_sentences.push_back(dt);
	}

	int64_t num_tokens = 0;
	for(auto& d : cnt_corpus) {
		if( counter.empty() ) {
			counter[d.first] = d.second;
			num_tokens += d.second;
		} else {
			if( slt_corpus.find(d.first) == slt_corpus.end()) {
				counter[d.first] = d.second;
				num_tokens += d.second;
			}
		}
	}
	/*
	std::cout << "num_tokens:  " << num_tokens << '\n';

	int i = 0;
	for (auto it = counter.begin(); it != counter.end(); it++) {
		std::cout << "[" << it->first  << " : " << it->second << "] ";
		i++;
		if( i > 5 )
			break;
	}
	std::cout << '\n';
	*/

	std::vector<std::vector<std::string>> n_sentences;

	for(int64_t r = 0; r < slt_sentences.size(); r++) {
		std::vector<std::string> dt;
		for(auto& d : slt_sentences[r]) {
			if( keep(d, counter, num_tokens) ) {
				dt.push_back(d);
			}
		}

		if(dt.size() > 0 ) {
			//printVector(dt);
			n_sentences.push_back(dt);
		}
	}

	return std::make_pair(n_sentences, counter);
}

std::pair<std::vector<int64_t>, std::vector<std::vector<int64_t>>> get_centers_and_contexts(
		std::vector<std::vector<int64_t>> corpus, int64_t max_window_size) {

	//std::cout << "corpus.size(): " << corpus.size() << '\n';
	std::default_random_engine e(static_cast<unsigned int>(time(nullptr)));
	std::uniform_int_distribution<> randInt(1, max_window_size);
    //Return center words and context words in skip-gram.
	std::vector<int64_t> centers;
	std::vector<std::vector<int64_t>> contexts;

    for(auto& line : corpus ) {
        // To form a "center word--context word" pair, each sentence needs to
        // have at least 2 words
    	//std::cout << "line.size(): " << line.size() << '\n';
        if(line.size() < 2 )
            continue;
        //centers += line
        for(auto& d : line )
        	centers.push_back(d);

        int64_t cnt = line.size(), start = 0;
        for(auto& i : range(cnt, start)) {  // Context window centered at `i`
            int64_t window_size = randInt(e); //random.randint(1, max_window_size)
            int64_t count = std::min(cnt, i + 1 + window_size) - std::max(start, (i - window_size));
            //indices = list(range(max(0, i - window_size),
            //                     min(len(line), i + 1 + window_size)))
            std::vector<int64_t> indices = range(count, std::max(start, (i - window_size)));
            //std::cout << "window_size: " << window_size
            //		  << " count: " << count << " i: " << i << " max: " << std::max(start, (i - window_size)) << '\n';
            //printVector(indices);
            // Exclude the center word from the context words
            indices.erase(std::remove(indices.begin(), indices.end(), i), indices.end());
            //printVector(indices);
            //contexts.append([line[idx] for idx in indices])
            std::vector<int64_t> dt;
            for(auto& idx : indices )
            	dt.push_back(line[idx]);

            contexts.push_back(dt);
        }
    }
    return std::make_pair(centers, contexts);
}

//return random double from low to high, default 0-1
double randomf(double low, double high) {
	return low + (high - low) * rand() / ((double) RAND_MAX);
}


std::vector<std::vector<int64_t>> get_negatives(std::vector<std::vector<int64_t>> all_contexts, Vocab vocab,
		std::map<std::string, int64_t> counter, int64_t K) {
    // Return noise words in negative sampling.
    // Sampling weights for words with indices 1, 2, ... (index 0 is the
    // excluded unknown token) in the vocabulary

	//std::cout << "all_contexts: " << all_contexts.size()
	//		  << " counter: " << counter.size()
	//		  << " vocab: " << vocab.length() << '\n';

	std::vector<double> sampling_weights;
	std::vector<std::vector<int64_t>>  all_negatives;
	int64_t cnt = vocab.length() - 1, start = 1;
	for(auto& i : range(cnt, start) ) {
		std::vector<int64_t> indices;
		indices.push_back(i);
		std::vector<std::string> dt = vocab.to_tokens(indices);
		for(auto& str : dt)
			sampling_weights.push_back(std::pow(static_cast<double>(counter[str]), 0.75));
	}
	//std::cout << "sampling_weights: " << sampling_weights.size() << " vocab.length(): " << vocab.length() << '\n';
    //sampling_weights = [counter[vocab.to_tokens(i)]**0.75
    RandomGenerator generator(sampling_weights);

    for(auto& contexts : all_contexts ) {
    	std::vector<int64_t> negatives;
        while( negatives.size() < contexts.size() * K ) {
            int64_t neg = generator.draw();
            // Noise words cannot be context words
            if(negatives.size() < 1) {
            	negatives.push_back(neg);
            } else {
            	if( std::find(contexts.begin(), contexts.end(), neg) == contexts.end() ) // neg not in contexts
            		negatives.push_back(neg);
            }
        }
        all_negatives.push_back(negatives);
    }

	return all_negatives;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>  batchify(
		std::vector<std::vector<int64_t>> all_contexts,
		std::vector<std::vector<int64_t>> all_negatives,
		std::vector<int64_t> all_centers) {
	int64_t max_len = 0;
	for( int i = 0; i < all_contexts.size(); i++ ) {
		std::vector<int64_t> d1 = all_contexts[i];
		std::vector<int64_t> d2 = all_negatives[i];
		if( (d1.size() + d2.size()) > max_len )
			max_len = d1.size() + d2.size();
	}
	//std::cout << "max_len: " << max_len << " all_contexts.size(): " << all_contexts.size() << '\n';

    std::vector<torch::Tensor> contexts_negatives, masks, labels;

    for( int64_t i = 0; i < all_contexts.size(); i++ ) {
    	std::vector<int64_t> d1 = all_contexts[i];
    	std::vector<int64_t> d2 = all_negatives[i];
        int64_t cur_len = d1.size() + d2.size();
        std::vector<int64_t> c_n;
        for(auto& t : d1)
        	c_n.push_back(t);
        for(auto& t : d2)
            c_n.push_back(t);
        //c_n.resize(cur_len);
        //std::merge(d1.begin(), d1.end(), d2.begin(), d2.end(), c_n.begin());
        for(int64_t j = 0; j < (max_len - cur_len); j++)
        	c_n.push_back(0);
        //if(c_n.size() > 60) std::cout << "c_n: " << c_n.size() << '\n';

        auto TT = torch::from_blob(c_n.data(), {1, static_cast<long>(c_n.size())}, torch::TensorOptions(torch::kLong));
        contexts_negatives.push_back(TT.clone());

        std::vector<int64_t> msk;
        for(int64_t j = 0; j < cur_len; j++)
        	msk.push_back(1);
        for(int64_t j = 0; j < (max_len - cur_len); j++)
        	msk.push_back(0);
        //if(msk.size() > 60) std::cout << "msk: " << msk.size() << '\n';

        auto mk = torch::from_blob(msk.data(), {1, static_cast<long>(msk.size())}, torch::TensorOptions(torch::kLong));
        masks.push_back(mk.clone());

        std::vector<int64_t> lbl;
        for(int64_t j = 0; j < d1.size(); j++)
        	lbl.push_back(1);

        for(int64_t j = 0; j < (max_len - d1.size()); j++)
            lbl.push_back(0);
        //if(lbl.size() > 60) std::cout << "lbl: " << lbl.size() << '\n';

        auto lb = torch::from_blob(lbl.data(), {1, static_cast<long>(lbl.size())}, torch::TensorOptions(torch::kLong));
        labels.push_back(lb.clone());
    }

    torch::Tensor contexts_negatives_ts = torch::concat(contexts_negatives, 0).to(torch::kLong);
    torch::Tensor masks_ts = torch::concat(masks, 0).to(torch::kLong);
    torch::Tensor labels_ts = torch::concat(labels, 0).to(torch::kLong);
    torch::Tensor centers_ts = torch::from_blob(all_centers.data(),
    			 { static_cast<long>(all_centers.size()), 1}, torch::TensorOptions(torch::kLong)).clone();

    return std::make_tuple(contexts_negatives_ts, masks_ts, labels_ts, centers_ts);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Vocab> load_data_ptb(
		std::string file_dir, int64_t batch_size, int64_t max_window_size,
		int64_t num_noise_words, int64_t num_samples) {

	std::vector<std::vector<std::string>> ptb_data = read_ptb(file_dir, num_samples);
	std::vector<std::string> tokens;

	for(int64_t i = 0; i < ptb_data.size(); i++ ) {
		std::vector<std::string> dt = ptb_data[i];
		for(int64_t j = 0; j < dt.size(); j++ )
			tokens.push_back(dt[j]);
	}

	std::vector<std::pair<std::string, int64_t>> cnt_corpus = count_corpus( tokens );

	float min_freq = 10.0f;
	std::vector<std::string> reserved_tokens(0);

	Vocab vocab(cnt_corpus, min_freq, reserved_tokens);

	std::pair<std::vector<std::vector<std::string>>, std::map<std::string, int64_t>> rlt = subsample(
			ptb_data, vocab, cnt_corpus);
	std::vector<std::vector<std::string>> subsampled = rlt.first;
	std::map<std::string, int64_t> counter = rlt.second;

	std::vector<std::string> tks;
	for(auto& dt: subsampled ) {
		for(auto& d : dt)
			tks.push_back(d);
	}

	std::vector<std::pair<std::string, int64_t>> aft_freq = count_corpus( tks );

	std::vector<std::vector<int64_t>> corpus;
	for(auto& line : subsampled) {
		corpus.push_back(vocab[line]);
	}

	std::vector<int64_t> all_centers;
	std::vector<std::vector<int64_t>> all_contexts;
	std::tie(all_centers, all_contexts) = get_centers_and_contexts(corpus, max_window_size);

	std::vector<std::vector<int64_t>> all_negatives = get_negatives(all_contexts, vocab, counter, num_noise_words);
	int64_t max_len = 0;
	for( int i = 0; i < all_contexts.size(); i++ ) {
		std::vector<int64_t> d1 = all_contexts[i];
		std::vector<int64_t> d2 = all_negatives[i];
		if( (d1.size() + d2.size()) > max_len )
			max_len = d1.size() + d2.size();
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> batch = batchify(
			all_contexts, all_negatives, all_centers);
	// contexts_negatives_ts, masks_ts, labels_ts, centers_ts
	return std::make_tuple(std::get<0>(batch), std::get<1>(batch), std::get<2>(batch),std::get<3>(batch), vocab);
}

