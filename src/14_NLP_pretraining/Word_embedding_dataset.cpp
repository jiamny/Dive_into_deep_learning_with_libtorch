
#include "../utils/ch_14_util.h"
#include <matplot/matplot.h>
using namespace matplot;

void show_list_len_pair_hist(std::vector<std::vector<std::string>> a, std::vector<std::vector<std::string>> b) {
	std::vector<double> org, sub;
	for(auto&d : a)
		org.push_back(d.size()*1.0);
	for(auto&d : b)
		sub.push_back(d.size()*1.0);

    auto h1 = matplot::hist(org, 10);
    hold(on);
    auto h2 = matplot::hist(sub, 10);
    h1->normalization(histogram::normalization::probability).display_name("origin");;
    h1->bin_width(0.5);
    h2->normalization(histogram::normalization::probability).display_name("subsampled");;
    h2->bin_width(0.5);
    xlabel("# tokens per sentence");
    ylabel("Count");
    legend({});
    show();
}


std::string compare_counts(std::string token, Vocab a, std::vector<std::pair<std::string, int64_t>> aft_freq) {

	std::vector<std::pair<std::string, int64_t>> org_freq = a.token_freqs();
	int64_t o_freq = 0, a_freq = 0;
	for(auto& d : org_freq) {
		if(d.first == token) {
			o_freq = d.second;
			break;
		}
	}
	for(auto& d : aft_freq) {
		if(d.first == token) {
			a_freq = d.second;
			break;
		}
	}

    return "# of [" + token + "] before=" + std::to_string(o_freq) + ", after=" + std::to_string(a_freq);
}


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Reading the Dataset\n";
	std::cout << "// -----------------------------------------------------------------\n";

	const std::string file_dir = "./data/ptb";

	std::vector<std::vector<std::string>> ptb_data = read_ptb(file_dir, 0);
	std::vector<std::string> tokens;

	for(int i = 0; i < 5; i++ )
		printVector(ptb_data[i]);

	for(int64_t i = 0; i < ptb_data.size(); i++ ) {
		std::vector<std::string> dt = ptb_data[i];
		for(int64_t j = 0; j < dt.size(); j++ )
			tokens.push_back(dt[j]);
	}

	std::vector<std::pair<std::string, int64_t>> cnt_corpus = count_corpus( tokens );

	float min_freq = 10.0f;
	std::vector<std::string> reserved_tokens(0);

	Vocab vocab(cnt_corpus, min_freq, reserved_tokens);
	std::cout << "vocab.length: " << vocab.length() << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Subsampling\n";
	std::cout << "// -----------------------------------------------------------------\n";

	std::pair<std::vector<std::vector<std::string>>, std::map<std::string, int64_t>> rlt = subsample(
			ptb_data, vocab, cnt_corpus);
	std::vector<std::vector<std::string>> subsampled = rlt.first;
	std::map<std::string, int64_t> counter = rlt.second;

	show_list_len_pair_hist(ptb_data, subsampled);

	for(int i = 0; i < 5; i++ )
		printVector(subsampled[i]);

	std::vector<std::string> tks;
	for(auto& dt: subsampled ) {
		for(auto& d : dt)
			tks.push_back(d);
	}

	std::vector<std::pair<std::string, int64_t>> aft_freq = count_corpus( tks );

	std::string token = "the";
	std::cout << compare_counts(token, vocab, aft_freq) << '\n';
	token = "join";
	std::cout << compare_counts(token, vocab, aft_freq) << '\n';

	std::vector<std::vector<int64_t>> corpus;
	for(auto& line : subsampled) {
		corpus.push_back(vocab[line]);
	}

	for(int i = 0; i < 3; i++)
		printVector(corpus[i]);

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Extracting Center Words and Context Words\n";
	std::cout << "// -----------------------------------------------------------------\n";
	std::vector<std::vector<int64_t>> tiny_dataset = {{0, 1, 2, 3, 4, 5, 6}, {7, 8, 9}};
	std::cout << "dataset:\n";
	for(auto& d : tiny_dataset)
		printVector(d);

	std::vector<int64_t> centers;
	std::vector<std::vector<int64_t>> contexts;

	std::tie(centers, contexts) = get_centers_and_contexts(tiny_dataset, 2);

	for(int d = 0; d < contexts.size(); d++) {
		std::cout << "center: " << centers[d] << " has contexts: ";
		printVector(contexts[d]);
	}
	std::cout << "\n";

	std::vector<int64_t> all_centers;
	std::vector<std::vector<int64_t>> all_contexts;
	std::tie(all_centers, all_contexts) = get_centers_and_contexts(corpus, 5);
	int64_t sm = 0;
	for(auto& c : all_contexts )
		sm += c.size();

	std::cout << "# center-context pairs: " << sm << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Negative Sampling\n";
	std::cout << "// -----------------------------------------------------------------\n";

	std::vector<std::vector<int64_t>> all_negatives = get_negatives(all_contexts, vocab, counter, 5);
	std::cout << "After get_negatives\n";

	int64_t max_len = 0;
	for( int i = 0; i < all_contexts.size(); i++ ) {
		std::vector<int64_t> d1 = all_contexts[i];
		std::vector<int64_t> d2 = all_negatives[i];
		if( (d1.size() + d2.size()) > max_len )
			max_len = d1.size() + d2.size();
	}
	std::cout << "max_len: " << max_len << " all_contexts.size(): " << all_contexts.size() << '\n';

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Loading Training Examples\n";
	std::cout << "// -----------------------------------------------------------------\n";
	std::vector<std::vector<int64_t>> x_1 = {{2, 2}, {2, 2, 2}};
	std::vector<std::vector<int64_t>> x_2 = {{3, 3, 3, 3},{3, 3}};
	std::vector<int64_t> cnts = {1, 1};
	//x_1 = (1, [2, 2], [3, 3, 3, 3])
	//x_2 = (1, [2, 2, 2], [3, 3])
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> batch = batchify(x_1, x_2, cnts);

	std::vector<std::string> names = {"contexts_negatives", "masks", "labels", "centers"};
	for( int i = 0; i < names.size(); i++ ) {
		torch::Tensor dt;
		if(i == 0 ) dt = std::get<0>(batch);
		if(i == 1 ) dt = std::get<1>(batch);
		if(i == 2 ) dt = std::get<2>(batch);
		if(i == 3 ) dt = std::get<3>(batch);
		std::cout << "name: " << names[i] << " " << dt << '\n';
	}

	std::cout << "// -----------------------------------------------------------------\n";
	std::cout << "// Putting It All Together\n";
	std::cout << "// -----------------------------------------------------------------\n";
	// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, Vocab>
	// contexts_negatives_ts, masks_ts, labels_ts, centers_ts
	int64_t batch_size = 512, max_window_size = 5, num_noise_words = 5;
	torch::Tensor contexts_negatives_ts, masks_ts, labels_ts, centers_ts;

	std::tie(contexts_negatives_ts, masks_ts, labels_ts, centers_ts, vocab) =  load_data_ptb(
			file_dir, batch_size, max_window_size, num_noise_words);

	auto dataset = PTBDataset(centers_ts).map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
			        					std::move(dataset),
										torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	for(auto& batch_data : *train_iter) {
		torch::Tensor centers  = batch_data.data;
		torch::Tensor tidx  = batch_data.target.squeeze();

		torch::Tensor ctx_neg_ts = torch::index_select(contexts_negatives_ts, 0, tidx);
	    torch::Tensor msk_ts = torch::index_select(masks_ts, 0, tidx);
	    torch::Tensor lbls     = torch::index_select(labels_ts, 0, tidx).flatten();

	    std::cout << "contexts_negatives_ts: " << ctx_neg_ts.sizes()
	    		  << " masks_ts: " << msk_ts.sizes()
				  << " labels_ts: " << lbls.sizes()
				  << " centers_ts: " << centers.sizes()<< '\n';
	    break;
	}

	std::cout << "Done\n";
	return 0;
}

