
#include <torch/utils.h>
#include <torch/torch.h>
#include "../utils/ch_8_9_util.h"
#include "../utils/ch_14_util.h"
#include "../utils/ch_15_util.h"


class SNLIBERTDataset : public torch::data::datasets::Dataset<SNLIBERTDataset> {
    //A customized dataset to load the SNLI dataset
private:
    int64_t max_len;
    Vocab vocab;
    std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> all_premise_hypothesis_tokens;
    torch::Tensor all_token_ids, all_segments, valid_lens, labels, data_idx;
public:

	explicit SNLIBERTDataset(std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<int64_t>>
			dataset, int64_t max_len, Vocab vocab) {
        this->max_len = max_len;
        std::vector<std::string> all_premise_tokens = std::get<0>(dataset);
        std::vector<std::string> all_hypothesis_tokens = std::get<1>(dataset);
        std::vector<int64_t> idx;
        int sz = all_premise_tokens.size();
        for(int i = 0; i < sz; i++) {
        	std::string s = all_premise_tokens[i];
        	std::string ss = all_hypothesis_tokens[i];
        	std::transform(s.begin(), s.end(), s.begin(), ::tolower);
        	std::transform(ss.begin(), ss.end(), ss.begin(), ::tolower);
        	std::vector<std::string> tks = tokenize_str(s);
        	std::vector<std::string> tkss = tokenize_str(ss);
        	all_premise_hypothesis_tokens.push_back(std::make_pair(tks, tkss));
        	idx.push_back(i);
        }

        std::vector<int64_t> labels_dt = std::get<2>(dataset);
        labels = torch::from_blob(labels_dt.data(),
        							 {static_cast<long>(labels_dt.size()), 1}, torch::TensorOptions(torch::kLong)).clone();
        labels.to(torch::kLong);

        if( vocab.length() == 0 ) {
        	float min_freq = 5.0f;
        	std::vector<std::string> reserved_tokens;
        	reserved_tokens.push_back("<pad>");

            this->vocab = get_snil_vocab( dataset, min_freq, reserved_tokens);

        } else {
            this->vocab = vocab;
        }

        _preprocess(all_premise_hypothesis_tokens);

        data_idx = torch::from_blob(idx.data(),
        							 {static_cast<long>(idx.size()), 1}, torch::TensorOptions(torch::kLong)).clone();
        data_idx.to(torch::kLong);

	}

	void _preprocess(
			std::vector<std::pair<std::vector<std::string>, std::vector<std::string>>> all_premise_hypothesis_tokens) {

		std::vector<torch::Tensor> tk_ids, segs;
		std::vector<int64_t> v_lens;
		for(int i = 0; i < all_premise_hypothesis_tokens.size(); i++) {
			std::vector<std::string> p_tks, h_tks, tokens;
			std::vector<int64_t> segments;
			std::tie(p_tks, h_tks) = all_premise_hypothesis_tokens[i];
			_truncate_pair_of_tokens(p_tks, h_tks);
			std::tie(tokens, segments) = get_tokens_and_segments(p_tks, h_tks);

			int64_t valid_len = tokens.size();
			std::vector<int64_t> token_ids = vocab[tokens];
			int64_t dif = max_len - valid_len;

			if( dif > 0 ) {
				for(int64_t r = 0; r < dif; r++)
					token_ids.push_back(vocab["pad"]);
			}

			dif = max_len - segments.size();
			if( dif > 0 ) {
				for(int64_t r = 0; r < dif; r++)
					segments.push_back(0);
			}

			tk_ids.push_back( torch::from_blob(token_ids.data(), {1, static_cast<long>(max_len)},
					   torch::TensorOptions(torch::kLong)).clone() );

			segs.push_back( torch::from_blob(segments.data(), {1, static_cast<long>(max_len)},
					   torch::TensorOptions(torch::kLong)).clone() );

			v_lens.push_back(valid_len);
		}

		all_token_ids = torch::concat(tk_ids, 0).to(torch::kLong);
		all_segments = torch::concat(segs, 0).to(torch::kLong);
		valid_lens = torch::from_blob(v_lens.data(),
		        							 {static_cast<long>(v_lens.size()), 1}, torch::TensorOptions(torch::kLong)).clone();
	}

    void _truncate_pair_of_tokens(std::vector<std::string>& p_tokens, std::vector<std::string>& h_tokens) {
        // Reserve slots for '<CLS>', '<SEP>', and '<SEP>' tokens for the BERT
        // input
        while((p_tokens.size() + h_tokens.size()) > max_len - 3 ) {
            if( p_tokens.size() > h_tokens.size())
                p_tokens.pop_back();
            else
                h_tokens.pop_back();
        }
    }

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getData(void) {
    	return std::make_tuple(all_segments, valid_lens, labels);
    }

    torch::data::Example<> get(size_t idx) override {
        return {all_token_ids[idx], data_idx[idx]};
    }

    torch::optional<size_t> size() const override {
        return all_token_ids.size(0);
    }
};

struct BERTClassifierImpl : public torch::nn::Module {
	BERTEncoder encoder{nullptr};
	torch::nn::Linear output{nullptr};
	torch::nn::Sequential hidden{nullptr};

	BERTClassifierImpl( BERTModel bert) {
        encoder = bert->encoder;
        hidden = bert->hidden;
        output = torch::nn::Linear(torch::nn::LinearOptions(256, 3)); //nn.Dense(256, 3);
        register_module("encoder", encoder);
        register_module("hidden", hidden);
        register_module("output", output);
	}

    torch::Tensor forward( std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> inputs) {
    	torch::Tensor tokens_X = std::get<0>(inputs),
    					segments_X = std::get<1>(inputs),
						valid_lens_x = std::get<2>(inputs);
    	torch::Tensor encoded_X = encoder(tokens_X, segments_X, valid_lens_x.reshape(-1)); //valid_lens_x.squeeze(1));
        return output->forward(hidden->forward(encoded_X.index({Slice(), 0, Slice()})));
    }
};
TORCH_MODULE(BERTClassifier);


int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	std::string file_name = "./data/bert.small.torch/vocab.json";

	std::string line;
	std::ifstream fL;
	std::vector<std::pair<std::string, int64_t>> tk_freq;

	fL.open(file_name.c_str());
	std::cout << fL.is_open() << '\n';

	std::string delimiter = ", ";
	const char * pStr1 = NULL;
	const char * pStr2 = NULL;
	pStr1 = "[";
	pStr2 = "]";

	if( fL.is_open() ) {
		std::getline(fL, line);
		size_t rpos = line.find(pStr1);

		if(rpos >= 0 )
			line.replace(rpos, 1, "");

		rpos = line.find(pStr2);
		if(rpos >= 0 )
			line.replace(rpos, 1, "");

		size_t pos = 0;

		while ((pos = line.find(delimiter)) != std::string::npos) {
			std::string s = line.substr(0, pos);
			s = s.substr(1, s.length() - 2);
			if(s.length() > 0 )
				tk_freq.push_back(std::make_pair(s, 1));

			line.erase(0, pos + delimiter.length());
			//std::cout << s << '\n';
		}
		std::string s = line.substr(0, pos);
		s = s.substr(1, s.length() - 2);
		if(s.length() > 0 )
			tk_freq.push_back(std::make_pair(s, 1));
	}

	fL.clear();
	fL.close();

	Vocab vocab(tk_freq);
	std::cout << "0 : " << vocab.idx_to_token[0] << '\n';
	std::cout << "1 : " << vocab.idx_to_token[1] << '\n';
	std::cout << "<pad> : " << vocab["<pad>"] << '\n';

	std::vector<int64_t> norm_shape={256};

	int64_t num_hiddens=256, ffn_num_hiddens=512, num_heads=4, num_blks=2, dropout=0.1, max_len=128;
	int64_t ffn_num_input=256, num_layers=2, key_size=256, query_size=256,
            value_size=256, hid_in_features=256, mlm_in_features=256, nsp_in_features=256;

	BERTModel model = BERTModel(vocab.length(), num_hiddens, norm_shape, ffn_num_input,
				ffn_num_hiddens, num_heads, num_layers, dropout, max_len, key_size, query_size,
				value_size, hid_in_features, mlm_in_features, nsp_in_features, device);
	model->to(device);

	//Read the SNLI dataset into premises, hypotheses, and labels.
	const std::string data_dir = "./data/snli_1.0";
	const bool is_train = true;

	auto train_data = read_snli(data_dir, is_train, 0); // 0 - use all data

	for(int c = 0; c < 2; c++) {
		std::cout<< "0: " << std::get<0>(train_data)[c] << '\n';
		std::string s = std::get<0>(train_data)[c];
		std::transform(s.begin(), s.end(), s.begin(), (int (*) (int)) tolower); //[](unsigned char c){ return std::tolower(c); });
		std::cout << std::get<0>(train_data)[c] << " <=> " << s << '\n';
		std::vector<std::string> tks = tokenize_str(s);
		std::cout << "tks.length(): " << tks.size() << "\n";
		std::cout << std::get<1>(train_data)[c] << '\n';
		std::cout << std::get<2>(train_data)[c] << '\n';
	}

	auto test_data = read_snli(data_dir, false, 0);
	for(auto& data : {train_data, test_data}) {
		std::vector<int64_t> row = std::get<2>(data);
		std::cout << "[";
		for(int i = 0; i < 3; i++) {
			if( i < 2 )
				std::cout << std::count(row.begin(), row.end(), i) << ", ";
			else
				std::cout << std::count(row.begin(), row.end(), i);
		}
		std::cout << "]\n";
	}

	size_t batch_size = 16;
	SNLIBERTDataset train_set(train_data, max_len, vocab);

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> sdt = train_set.getData();
	torch::Tensor all_segments = std::get<0>(sdt).to(device);
	torch::Tensor valid_lens = std::get<1>(sdt).to(device);
	torch::Tensor labels = std::get<2>(sdt).to(device);
	std::cout << "all_segments: " << all_segments.sizes() << '\n';

	auto dataset = train_set.map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          std::move(dataset),
					  torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	SNLIBERTDataset test_set(test_data, max_len, vocab);
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> tdt = test_set.getData();
	torch::Tensor tall_segments = std::get<0>(tdt).to(device);
	torch::Tensor tvalid_lens = std::get<1>(tdt).to(device);
	torch::Tensor tlabels = std::get<2>(tdt).to(device);


	auto tdataset = test_set.map(torch::data::transforms::Stack<>());
	auto test_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			          std::move(tdataset),
					  torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	auto net = BERTClassifier(model);
	net->to(device);

	float lr =  1e-4;
	int num_epochs = 100;
	//auto trainer = torch::Ad nn::Adm nn.Adam(learning_rate=lr, params=net.trainable_params())
	torch::optim::Adam optimizer(net->parameters(), torch::optim::AdamOptions(lr));
	//loss = nn.CrossEntropyLoss(reduction="none")
	auto criterion = torch::nn::CrossEntropyLoss(torch::nn::CrossEntropyLossOptions().reduction(torch::kNone));


	for( int epoch = 0; epoch < num_epochs; epoch++ ) {
		net->train();
		printf("Epoch: %2d%s\n", (epoch + 1), "--------------------------------------------------------");
    	float loss_sum = 0.0;
    	size_t total_match = 0, total_counter = 0;
    	torch::Tensor responses;

		for(auto& dt : *train_iter ) {
			torch::Tensor token_ids = dt.data.to(device);
			torch::Tensor idx = dt.target.squeeze().to(device);
			size_t mini_batch_size = token_ids.size(0);

			torch::Tensor segments = torch::index_select(all_segments, 0, idx);
		    torch::Tensor val_lens = torch::index_select(valid_lens, 0, idx);
		    torch::Tensor lbls     = torch::index_select(labels, 0, idx).flatten();

		    torch::Tensor pred = net->forward(std::make_tuple(token_ids, segments, val_lens));

		    //torch::Tensor out = torch::nn::functional::log_softmax(pred, 1);
    		auto loss = criterion(pred, lbls); //torch::mse_loss(out, label)

    		optimizer.zero_grad();
    		loss.sum().backward();
    		optimizer.step();

    		total_counter += mini_batch_size;
    		loss_sum += loss.sum().item<float>();
    		total_match += accuracy(pred, lbls);
		}
		printf("train loss: %.3f train avg acc: %.3f\n", (loss_sum*1.0 / total_counter),
		      (total_match*1.0 / total_counter));

		net->eval();
		loss_sum = 0.0;
		total_match = 0, total_counter = 0;

		for(auto& dt : *test_iter ) {
			torch::Tensor token_ids = dt.data.to(device);
			torch::Tensor idx = dt.target.squeeze().to(device);
			size_t mini_batch_size = token_ids.size(0);

			torch::Tensor segments = torch::index_select(tall_segments, 0, idx);
		    torch::Tensor val_lens = torch::index_select(tvalid_lens, 0, idx);
		    torch::Tensor lbls     = torch::index_select(tlabels, 0, idx).flatten();

		    torch::Tensor pred = net->forward(std::make_tuple(token_ids, segments, val_lens));

    		total_counter += mini_batch_size;
    		total_match += accuracy(pred, lbls);
		}
		printf("validation avg acc: %.3f\n", (total_match*1.0 / total_counter));
	}

	std::cout << "Done!\n";
}


