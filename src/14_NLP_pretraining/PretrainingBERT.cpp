
#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include "../utils/ch_14_util.h"
#include "../TempHelpFunctions.hpp"

#include <matplot/matplot.h>
using namespace matplot;

template<typename T>
std::tuple <torch::Tensor, torch::Tensor, torch::Tensor>
_get_batch_loss_bert(T& net, torch::nn::CrossEntropyLoss loss, int64_t vocab_size, torch::Tensor tokens_X,
		torch::Tensor segments_X, torch::Tensor valid_lens_x,
		torch::Tensor pred_positions_X, torch::Tensor mlm_weights_X,
		torch::Tensor mlm_Y, torch::Tensor nsp_y) {
    // 前向传播
	std::cout << "前向传播\n";
	std::tuple <torch::Tensor, torch::Tensor, torch::Tensor> ft = net->forward(tokens_X, segments_X,
            valid_lens_x.reshape(-1), pred_positions_X);

    torch::Tensor _ = std::get<0>(ft), mlm_Y_hat = std::get<1>(ft), nsp_Y_hat = std::get<2>(ft);

    // 计算遮蔽语言模型损失
    std::cout << "计算遮蔽语言模型损失\n";
    mlm_Y = mlm_Y.to(torch::kLong);

    torch::Tensor mlm_l = loss(mlm_Y_hat.reshape({-1, vocab_size}), mlm_Y.reshape(-1)).mul(mlm_weights_X.reshape({-1, 1}));
    mlm_l = mlm_l.sum().div((mlm_weights_X.sum()).add(1e-8));

    // 计算下一句子预测任务的损失
    std::cout << "计算下一句子预测任务的损失\n";
    nsp_y = nsp_y.to(torch::kLong);
    torch::Tensor nsp_l = loss(nsp_Y_hat, nsp_y);
    std::cout << "after loss\n";
    torch::Tensor l = mlm_l.add( nsp_l );
    return std::make_tuple(mlm_l, nsp_l, l);
}

template<typename T>
void train_bert(_WikiTextDataset train_set, T& net, torch::nn::CrossEntropyLoss loss,
		int64_t vocab_size, int64_t num_steps, int64_t batch_size, torch::Device device) {

	std::cout << "Load data\n";
	auto dataset = train_set.map(torch::data::transforms::Stack<>());
	auto train_iter = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
				        	std::move(dataset),
							torch::data::DataLoaderOptions().batch_size(batch_size).drop_last(true));

	torch::optim::Adam trainer(net->parameters(), torch::optim::AdamOptions(0.01)); //lr

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
				   torch::Tensor, torch::Tensor> dts = train_set.getData();

	const torch::Tensor all_segments = std::get<0>(dts).to(device), valid_lens = std::get<1>(dts).to(device),
				   	    all_pred_positions = std::get<2>(dts).to(device), all_mlm_weights = std::get<3>(dts).to(device),
					  	all_mlm_labels = std::get<4>(dts).to(device), nsp_labels = std::get<5>(dts).to(device);

	int64_t step = 0;

    // 遮蔽语言模型损失的和，下一句预测任务损失的和
    bool num_steps_reached = false;

	std::vector<double> MLM_loss, NSP_loss;
	std::vector<double> steps;
	std::cout << "Start training...\n";
	net->train();
    while( step < num_steps && ! num_steps_reached ) {
        double mlm_t = 0.0, nsp_t = 0.0;
        int64_t nums = 0;
    	for(auto& dt : *train_iter ) {
    		torch::Tensor tokens_X = dt.data;
    		const torch::Tensor target = dt.target.to(device);

    		tokens_X = tokens_X.to(device);
    		torch::Tensor segments_X = torch::index_select(all_segments, 0, target.squeeze());
    		torch::Tensor valid_lens_x = torch::index_select(valid_lens, 0, target.squeeze());
    		torch::Tensor pred_positions_X = torch::index_select(all_pred_positions, 0, target.squeeze());
			torch::Tensor mlm_weights_X = torch::index_select(all_mlm_weights, 0, target.squeeze());
			torch::Tensor mlm_Y = torch::index_select(all_mlm_labels, 0, target.squeeze());
			torch::Tensor nsp_y = torch::index_select(nsp_labels, 0, target.squeeze());

            trainer.zero_grad();

            std::tuple <torch::Tensor, torch::Tensor, torch::Tensor> lbert = _get_batch_loss_bert(
                net, loss, vocab_size, tokens_X, segments_X, valid_lens_x,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_y);

			torch::Tensor mlm_l = std::get<0>(lbert), nsp_l = std::get<1>(lbert), l = std::get<2>(lbert);
            l.backward();
            trainer.step();

            step += 1;
            steps.push_back(step * 1.0);
            mlm_t = mlm_l.sum().data().item<double>();
            nsp_t = nsp_l.sum().data().item<double>();
            nums += tokens_X.size(0);
            MLM_loss.push_back( mlm_t );
            NSP_loss.push_back( nsp_t);
            std::cout << "Step: " << step << "; MLM loss: " << mlm_t << "; NSP loss : " << nsp_t << '\n';
            if( step == num_steps ) {
                num_steps_reached = true;
                break;
            }
    	}
    }
	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::plot(ax1, steps, MLM_loss, "g--")->line_width(2)
			.display_name("MLM loss");
	matplot::hold(ax1, true);
	matplot::plot(ax1, steps, NSP_loss, "r-.")->line_width(2)
			.display_name("NSP loss");
    matplot::xlabel(ax1, "step");
    matplot::ylabel(ax1, "loss");
    matplot::legend({});
    matplot::hold(ax1, false);
    matplot::show();
}

// 用BERT表示文本
template<typename T>
torch::Tensor get_bert_encoding(T& net, std::vector<std::string> tokens_a,
		std::vector<std::string> tokens_b, Vocab vocab, torch::Device device) {
	std::pair<std::vector<std::string>, std::vector<int64_t>> t = get_tokens_and_segments(tokens_a, tokens_b);
	std::vector<std::string> tokens =  t.first;
	std::vector<int64_t> segts = t.second;
	std::vector<int64_t> tk_ids = vocab[tokens];
	torch::Tensor token_ids = torch::from_blob(tk_ids.data(), {1, static_cast<int>(tk_ids.size())}, dtype(torch::kLong)).clone();
	torch::Tensor segments  = torch::from_blob(segts.data(), {1, static_cast<int>(segts.size())}, dtype(torch::kLong)).clone();
	torch::Tensor valid_len = torch::tensor({static_cast<int64_t>(tokens.size())}).to(torch::kLong).clone();
	token_ids = token_ids.to(device);
	segments = segments.to(device);
	valid_len = valid_len.to(device);

	std::cout << "token_ids===: " << token_ids.sizes() << " segments: " << segments.sizes()
			  << " valid_len: " <<  valid_len << '\n';
	std::tuple <torch::Tensor, torch::Tensor, torch::Tensor> ft = net->forward(token_ids, segments, valid_len);
	//torch::Tensor encoded_X
    return std::get<0>(ft);
}

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Device
	//torch::Device device(torch::kCPU);
	auto cuda_available = torch::cuda::is_available();
	torch::Device device(cuda_available ? torch::kCUDA : torch::kCPU);
	std::cout << (cuda_available ? "CUDA available. Training on GPU." : "Training on CPU.") << '\n';

	torch::manual_seed(123);

	const std::string data_dir = "./data/wikitext-2";
	std::vector<std::vector<std::vector<std::string>>> paragraphs = _read_wiki(data_dir, 3000); // 0 - use all data

	std::cout << paragraphs.size() << '\n';
	std::vector<std::vector<std::string>> lines = paragraphs[0];
	std::cout << lines.size() << '\n';
	std::vector<std::string> tks = lines[0];
	std::cout << tks.size() << '\n';
	printVector(tks);

	int64_t max_len = 64, batch_size = 512;

	_WikiTextDataset train_set(paragraphs, max_len);
	Vocab vocab = train_set.getVocab();
	std::cout << train_set.size().value() << '\n';

	// 预训练BERT
	std::cout << "预训练BERT\n";
	std::vector<int64_t> norm_shape = {128};
	BERTModel net = BERTModel( vocab.length(), 128, norm_shape, 128, 256, 2, 2, 0.2, 1000, 128, 128,
							   128, 128, 128, 128, device);
	net->to(device);
	/*
	    int64_t vocab_size, int64_t num_hiddens, std::vector<int64_t> norm_shape, int64_t ffn_num_input,
		int64_t ffn_num_hiddens, int64_t num_heads, int64_t num_layers, double dropout,
		int64_t max_len=1000, int64_t key_size=768, int64_t query_size=768, int64_t value_size=768,
		int64_t hid_in_features=768, int64_t mlm_in_features=768, int64_t nsp_in_features=768

		vocab.length(), int64_t num_hiddens=128, norm_shape,
		int64_t ffn_num_input=128, int64_t ffn_num_hiddens=256, int64_t num_heads=2,
		int64_t num_layers=2, double dropout=0.2, int64_t key_size=128, int64_t query_size=128,
		int64_t value_size=128, int64_t hid_in_features=128, int64_t mlm_in_features=128,
		int64_t nsp_in_features=128
	*/

	auto loss = torch::nn::CrossEntropyLoss();

	train_bert(train_set, net, loss, vocab.length(), 50, batch_size, device); // 50

	net->eval();
	std::cout << "用BERT表示文本\n";
	std::vector<std::string> tokens_a = {"a", "crane", "is", "flying"};
	std::vector<std::string> tokens_b;
	torch::Tensor encoded_text = get_bert_encoding(net, tokens_a, tokens_b, vocab, device);

	// 词元：'<cls>','a','crane','is','flying','<sep>'
	torch::Tensor encoded_text_cls = encoded_text.index({Slice(), 0, Slice()});
	torch::Tensor encoded_text_crane = encoded_text.index({Slice(), 2, Slice()});
	std::cout << "encoded_text: "<< encoded_text.sizes() << " encoded_text_cls: " << encoded_text_cls.sizes()
	    	  << " encoded_text_crane[0][:3]: " << encoded_text_crane.index({0, Slice(None,3)}) << '\n';

	tokens_a.clear();
	tokens_b.clear();
	tokens_a.push_back("a");
	tokens_a.push_back("crane");
	tokens_a.push_back("driver");
	tokens_a.push_back("came");
	tokens_b.push_back("he");
	tokens_b.push_back("just");
	tokens_b.push_back("left");

	torch::Tensor encoded_pair = get_bert_encoding(net, tokens_a, tokens_b, vocab, device);
	// 词元：'<cls>','a','crane','driver','came','<sep>','he','just','left','<sep>'
	torch::Tensor encoded_pair_cls = encoded_pair.index({Slice(), 0, Slice()});
	torch::Tensor encoded_pair_crane = encoded_pair.index({Slice(), 2, Slice()});
	std::cout << "encoded_pair: "<< encoded_pair.sizes() << " encoded_pair_cls: " << encoded_pair_cls.sizes()
		    	  << " encoded_pair_crane[0][:3]: " << encoded_pair_crane.index({0, Slice(None,3)}) << '\n';

	std::cout << "Done!\n";
}





