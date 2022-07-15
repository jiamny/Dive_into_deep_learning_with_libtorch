#include "ch_15_util.h"


std::pair<std::vector<std::string>, std::vector<int64_t>> read_imdb(std::string data_dir, bool is_train, int num_files) {
	//Read the IMDb review dataset text sequences and labels.
	std::vector<std::string> data;
	std::vector<int64_t> labels;
	std::vector<std::string> Labs = {"pos", "neg"};

	for(std::string& label : Labs) {
		std::string folder_name = data_dir;
		if( is_train ) {
		    folder_name += "/train/" + label;
		} else {
		    folder_name += "/test/" + label;
		}
		//std::cout << folder_name << '\n';
		int cnt = 0;
		for(auto& f : std::filesystem::directory_iterator(folder_name)) {
			//std::cout << f.path() << '\n';
		    std::string line;
		    std::ifstream fL(f.path().c_str());

		    if( fL.is_open() ) {
		    	while ( std::getline(fL, line) ) {
		    		//std::cout << line <<'\n';
		    		line = std::regex_replace(line, std::regex("\\\n"), "");
		    	    data.push_back(line);

		    		if( label == "pos" )
		    			labels.push_back(1);
		    		else
		    			labels.push_back(0);
		    	}
		    }

		    fL.close();
		    cnt++;
		    if( num_files > 0 ) {
		    	if( cnt > num_files ) break;
		    }
		}
	}

	return std::make_pair(data, labels);
}

std::pair<std::vector<std::string>, int> count_num_tokens(std::string text) {
	std::string space_delimiter = " ";
	std::vector<std::string> words{};
	int count = 0;

	size_t pos = 0;
	while ((pos = text.find(space_delimiter)) != std::string::npos) {
	    words.push_back(text.substr(0, pos));
		count++;
	    text.erase(0, pos + space_delimiter.length());
	}
	return std::make_pair(words, count);
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Vocab> load_data_imdb(
															std::string data_dir, size_t num_steps, int num_files) {
	// read in train data files
	bool is_train = true;
	auto acimdb = read_imdb(data_dir, is_train, num_files);
	auto data = acimdb.first;
	auto labels = acimdb.second;

	// read in test data files
	is_train = false;

	if(num_files > 0 ) num_files = static_cast<int>(num_files/2.0);

	auto tacimdb = read_imdb(data_dir, is_train, num_files);
	auto tdata = tacimdb.first;
	auto tlabels = tacimdb.second;

	//-------------------------------------------------------------
	// split words and extract first token upto max_tokens
	//-------------------------------------------------------------
	std::vector<std::string> ttokens = tokenize(data, "word", false);

	int64_t max_tokens = ttokens.size() - 1;

	std::vector<std::string> tokens(&ttokens[0], &ttokens[max_tokens]);

	std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

	std::vector<std::string> reserved_tokens;
	reserved_tokens.push_back("<pad>");
	Vocab vocab = Vocab(counter, 5.0, reserved_tokens);

	std::vector<torch::Tensor> tensors;

	for( int i = 0; i < data.size(); i++ ) {
		auto dt = truncate_pad(vocab[count_num_tokens(data[i]).first], num_steps, vocab["<pad>"]);

		auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();

		tensors.push_back(TT);
	}

	torch::Tensor features = torch::concat(tensors, 0).to(torch::kLong);
	torch::Tensor rlab = torch::from_blob(labels.data(),
						 {static_cast<long>(labels.size()), 1}, torch::TensorOptions(torch::kLong)).clone();

	rlab.to(torch::kLong);

	// test dataset
	std::vector<torch::Tensor> tstensors;

	for( int i = 0; i < tdata.size(); i++ ) {
		auto dt = truncate_pad(vocab[count_num_tokens(tdata[i]).first], num_steps, vocab["<pad>"]);

		auto TT = torch::from_blob(dt.data(), {1, static_cast<long>(num_steps)}, torch::TensorOptions(torch::kLong)).clone();

		tstensors.push_back(TT);
	}

	torch::Tensor tfeatures = torch::concat(tstensors, 0).to(torch::kLong);
	torch::Tensor trlab = torch::from_blob(tlabels.data(),
							 {static_cast<long>(tlabels.size()), 1}, torch::TensorOptions(torch::kLong)).clone();

	trlab.to(torch::kLong);


	return std::make_tuple(features, rlab, tfeatures, trlab, vocab);
}



