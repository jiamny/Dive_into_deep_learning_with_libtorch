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
	std::vector<std::string> tokens;
    tokens.push_back("<cls>");
    for(auto& s :tokens_a)
    	tokens.push_back(s);
    tokens.push_back("<sep>");

    // 0 and 1 are marking segment A and B, respectively
    //segments = [0] * (len(tokens_a) + 2)
    std::vector<int64_t> segments;
    for( int i = 0; i < (tokens_a.size() + 2); i++ )
    	segments.push_back(0);

    if( ! tokens_b.empty() ) {
        //tokens += tokens_b + ['<sep>']
    	for(auto& s :tokens_b)
    	    tokens.push_back(s);
    	tokens.push_back("<sep>");

        //segments += [1] * (len(tokens_b) + 1)
    	for( int i = 0; i < (tokens_b.size() + 1); i++ )
    	    	segments.push_back(1);
    }
    return std::make_pair(tokens, segments);
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


std::tuple<std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
		   std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>>
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
    std::vector<int64_t> token_ids, pred_positions, mlm_pred_label_ids, segments;
    bool is_next;
    for( auto& exp : examples ) {
        token_ids = std::get<0>(exp);
        pred_positions = std::get<1>(exp);
        mlm_pred_label_ids = std::get<2>(exp);
        segments = std::get<3>(exp);
        is_next = std::get<4>(exp);

        for(size_t i = 0; i < (max_len - token_ids.size()); i++ )
        token_ids.push_back(vocab["<pad"]);

        torch::Tensor t = torch::from_blob(token_ids.data(), {1, static_cast<int>(token_ids.size())}, dtype(torch::kLong)).clone();
        all_token_ids.push_back(t.clone());

        for(size_t i = 0; i < (max_len - segments.size()); i++ )
        	segments.push_back(0);

        t = torch::from_blob(segments.data(), {1, static_cast<int>(segments.size())}, dtype(torch::kLong)).clone();
        all_segments.push_back(t.clone());

        valid_lens.push_back(torch::full({1}, static_cast<int>(token_ids.size())).to(torch::kFloat32).clone());

        for(size_t i = 0; i < (max_num_mlm_preds - pred_positions.size()); i++ )
        	pred_positions.push_back(0);

        t = torch::from_blob(pred_positions.data(), {1, static_cast<int>(pred_positions.size())}, dtype(torch::kLong)).clone();
        all_pred_positions.push_back(t.clone());

        // Predictions of padded tokens will be filtered out in the loss via
        // multiplication of 0 weights

        std::vector<float> vt;
        for(int i = 0; i < mlm_pred_label_ids.size(); i++)
        	vt.push_back(1.0f);

        for(int i = 0; i < (max_num_mlm_preds - pred_positions.size()); i++)
            vt.push_back(0.0f);

        t = torch::from_blob(vt.data(), {1, static_cast<int>(vt.size())}, dtype(torch::kFloat32)).clone();
        all_mlm_weights.push_back(t.clone());


        for( int i = 0; i < (max_num_mlm_preds - mlm_pred_label_ids.size()); i++ )
        	mlm_pred_label_ids.push_back(0);

        t = torch::from_blob(mlm_pred_label_ids.data(), {1, static_cast<int>(mlm_pred_label_ids.size())}, dtype(torch::kLong)).clone();
        all_mlm_labels.push_back(t.clone());

		if( is_next )
			nsp_labels.push_back(torch::full({1}, 1).to(torch::kLong).clone());
		else
			nsp_labels.push_back(torch::full({1}, 0).to(torch::kLong).clone());

    }
    return std::make_tuple(all_token_ids, all_segments, valid_lens, all_pred_positions,
            all_mlm_weights, all_mlm_labels, nsp_labels);
}


