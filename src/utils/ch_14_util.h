
#ifndef SRC_UTILS_CH_14_UTIL_H_
#define SRC_UTILS_CH_14_UTIL_H_
#pragma once
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <torch/nn.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <regex>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <map>
#include <set>
#include <random>
#include <iostream>
#include <filesystem>
#include <fstream>

#include "../utils/ch_8_9_util.h"
#include "../utils.h"

namespace F = torch::nn::functional;
using torch::indexing::Slice;
using torch::indexing::None;
using torch::indexing::Ellipsis;

std::string strip( const std::string& s );

std::vector<std::vector<std::vector<std::string>>> _read_wiki(const std::string data_dir, size_t num_read=0);

std::pair<std::vector<std::string>, std::vector<int64_t>> get_tokens_and_segments(std::vector<std::string> tokens_a,
																				std::vector<std::string> tokens_b);

std::tuple<std::vector<std::string>, std::vector<std::string>, bool> _get_next_sentence(std::vector<std::string> sentence,
							std::vector<std::string> next_sentence, std::vector<std::vector<std::vector<std::string>>> paragraphs);

std::vector<std::tuple<std::vector<std::string>, std::vector<int64_t>, bool>> _get_nsp_data_from_paragraph(
		std::vector<std::vector<std::string>> paragraph, std::vector<std::vector<std::vector<std::string>>> paragraphs,
		Vocab vocab, size_t max_len);

std::pair<std::vector<std::string>, std::map<int64_t, std::string> > _replace_mlm_tokens(std::vector<std::string> tokens,
		std::vector<int64_t> candidate_pred_positions, int64_t num_mlm_preds, Vocab vocab);

std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>> _get_mlm_data_from_tokens(
		std::vector<std::string> tokens, Vocab vocab);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
		   torch::Tensor, torch::Tensor, torch::Tensor>
_pad_bert_inputs(std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
							 std::vector<int64_t>, std::vector<int64_t>, bool>> examples, size_t max_len, Vocab vocab);


torch::Tensor  transpose_output(torch::Tensor X, int64_t num_heads);

torch::Tensor transpose_qkv(torch::Tensor X, int64_t num_heads);


class _WikiTextDataset : public torch::data::datasets::Dataset<_WikiTextDataset> {
public:

	explicit _WikiTextDataset(std::vector<std::vector<std::vector<std::string>>> paragraphs, int64_t max_len) {
        // Input `paragraphs[i]` is a list of sentence strings representing a
        // paragraph; while output `paragraphs[i]` is a list of sentences
        // representing a paragraph, where each sentence is a list of tokens
		//std::vector<std::vector<std::string>> setences;
		std::vector<std::string> tokens;

		for(auto& paragraph : paragraphs) {
			for(auto& setence : paragraph) {

				for(auto& tk : setence)
					tokens.push_back(tk);

				//setences.push_back(setence);
			}
		}
		std::cout << "tokens: " << tokens.size() << '\n';
		std::cout << "tokens[0]: " << tokens[0] << '\n';

		std::vector<std::pair<std::string, int64_t>> counter = count_corpus( tokens );

		std::vector<std::string> reserved_tokens;
		reserved_tokens.push_back("<pad>");
		reserved_tokens.push_back("<mask>");
		reserved_tokens.push_back("<cls>");
		reserved_tokens.push_back("<sep>");
		vocab = Vocab(counter, 5.0, reserved_tokens);

		std::cout << "the: " << vocab["the"] << "\n";

	    std::vector<std::tuple<std::vector<int64_t>, std::vector<int64_t>,
									 std::vector<int64_t>, std::vector<int64_t>, bool>> examples;
	    for( auto& paragraph : paragraphs ) {
	    	// Get data for the next sentence prediction task

	    	auto nsp_data = _get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, max_len);

	    	for( auto& nsp : nsp_data ) {
	    		std::vector<std::string> tks;
	    		std::vector<int64_t>     segments;
	    		bool is_next;
	    		tks = std::get<0>(nsp);
				segments = std::get<1>(nsp);
				is_next = std::get<2>(nsp);
	    		// Get data for the masked language model task
	    		std::vector<int64_t> token_ids, pred_positions, mlm_pred_label_ids;
	    		std::tie(token_ids, pred_positions, mlm_pred_label_ids) = _get_mlm_data_from_tokens(tks, vocab);

	    		examples.push_back(std::make_tuple(token_ids, pred_positions, mlm_pred_label_ids, segments, is_next));
	    	}
	    }

        // Pad inputs
	    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
				   torch::Tensor, torch::Tensor, torch::Tensor>
	    rlt = _pad_bert_inputs(examples, max_len, vocab);
        all_token_ids = std::get<0>(rlt);
        all_segments = std::get<1>(rlt);
		valid_lens = std::get<2>(rlt);
        all_pred_positions = std::get<3>(rlt);
		all_mlm_weights = std::get<4>(rlt);
        all_mlm_labels = std::get<5>(rlt);
		nsp_labels = std::get<6>(rlt);
		tidxs = std::get<7>(rlt);
	}

    torch::data::Example<> get(size_t idx) override {
		return {all_token_ids[idx], tidxs[idx]};
	}

	torch::optional<size_t> size() const override {
        return all_token_ids.size(0);
	}

	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
			   torch::Tensor, torch::Tensor> getData(void) {
		return std::make_tuple(
		    		all_segments, valid_lens, all_pred_positions, all_mlm_weights,
					all_mlm_labels, nsp_labels);
	}

	Vocab getVocab(void) {
		return vocab;
	}
private:
	torch::Tensor all_token_ids, all_segments, valid_lens, all_pred_positions, all_mlm_weights,
				  all_mlm_labels, nsp_labels, tidxs;
	Vocab vocab;
};



struct AddNormImpl : torch::nn::Module {
    //The residual connection followed by layer normalization.
	AddNormImpl(std::vector<int64_t> norm_shape, double dp, torch::Device device=torch::kCPU) {
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(dp));
        ln = torch::nn::LayerNorm(torch::nn::LayerNormOptions(norm_shape));
        dropout->to(device);
        ln->to(device);
	}

	torch::Tensor forward(torch::Tensor X, torch::Tensor Y) {
        return ln->forward(dropout->forward(Y).add(X));
    }

	torch::nn::Dropout dropout{nullptr};
	torch::nn::LayerNorm ln{nullptr};
};
TORCH_MODULE(AddNorm);

struct PositionWiseFFNImpl : public torch::nn::Module {
    //The positionwise feed-forward network.
	PositionWiseFFNImpl(int64_t ffn_num_input, int64_t ffn_num_hiddens,
			int64_t ffn_num_outputs, torch::Device device=torch::kCPU) {
        dense1 = torch::nn::Linear(torch::nn::LinearOptions(ffn_num_input, ffn_num_hiddens).bias(false));
        relu = torch::nn::ReLU();
        dense2 = torch::nn::Linear(torch::nn::LinearOptions(ffn_num_hiddens, ffn_num_outputs).bias(false));
        dense1->to(device);
		relu->to(device);
		dense2->to(device);
	}

	torch::Tensor forward(torch::Tensor X) {
        return dense2->forward(relu->forward(dense1->forward(X)));
	}

	torch::nn::Linear dense1{nullptr}, dense2{nullptr};
	torch::nn::ReLU relu{nullptr};
};
TORCH_MODULE(PositionWiseFFN);

struct DotProductAttentionImpl : public torch::nn::Module {
    //缩放点积注意力
	DotProductAttentionImpl(double dp, torch::Device device=torch::kCPU) {
        dropout = torch::nn::Dropout(torch::nn::DropoutOptions(1 - dp));
        dropout->to(device);
	}

	torch::Tensor forward( torch::Tensor queries, torch::Tensor keys,
			torch::Tensor values, torch::Tensor valid_lens) {
        int64_t d = queries.size(-1);
        torch::Tensor scores = torch::bmm(queries, keys.swapaxes(1, 2)).div(std::sqrt(d*1.0)).to(queries.device());
        		//c10::Scalar(std::sqrt(d*1.0)));
        torch::Tensor attention_weights = masked_softmax(scores, valid_lens);
        return torch::bmm(dropout->forward(attention_weights), values);
	}
	torch::nn::Dropout dropout{nullptr};
};
TORCH_MODULE(DotProductAttention);


struct MultiHeadAttentionImpl : public torch::nn::Module {
    //多头注意力
	MultiHeadAttentionImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_hiddens,
			int64_t n_heads, double dp, bool has_bias=false, torch::Device device=torch::kCPU) {
        num_heads = n_heads;
        attention = DotProductAttention(dp, device);
        W_q = torch::nn::Linear(torch::nn::LinearOptions(query_size, num_hiddens).bias(has_bias));
        W_q->to(device);
        //Dense(query_size, num_hiddens, has_bias=has_bias)
		W_k = torch::nn::Linear(torch::nn::LinearOptions(key_size, num_hiddens).bias(has_bias));
		W_k->to(device);
		//Dense(key_size, num_hiddens, has_bias=has_bias)
		W_v = torch::nn::Linear(torch::nn::LinearOptions(value_size, num_hiddens).bias(has_bias));
		W_v->to(device);
		//Dense(value_size, num_hiddens, has_bias=has_bias)
		W_o = torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, num_hiddens).bias(has_bias));
		W_o->to(device);
		//Dense(num_hiddens, num_hiddens, has_bias=has_bias)
	}

    torch::Tensor forward(torch::Tensor queries, torch::Tensor keys, torch::Tensor values, torch::Tensor valid_lens) {
        queries = transpose_qkv(W_q->forward(queries), num_heads);
        keys    = transpose_qkv(W_k->forward(keys), num_heads);
        values  = transpose_qkv(W_v->forward(values), num_heads);
       
	   // valid_lens is not None:
        if(valid_lens.numel() != 0) {
        	valid_lens = torch::repeat_interleave(valid_lens, num_heads).to(queries.device());
        }

        auto output = attention->forward(queries, keys, values, valid_lens);
        torch::Tensor output_concat = transpose_output(output, num_heads);
        return W_o->forward(output_concat);
    }

    int64_t num_heads;
    torch::nn::Linear W_q{nullptr}, W_k{nullptr}, W_v{nullptr}, W_o{nullptr};
    DotProductAttention attention{nullptr};
};
TORCH_MODULE(MultiHeadAttention);


struct TransformerEncoderBlockImpl : public torch::nn::Module {
    //The Transformer encoder block.
	TransformerEncoderBlockImpl(int64_t key_size, int64_t query_size, int64_t value_size, int64_t num_hiddens,
			std::vector<int64_t> norm_shape, int64_t ffn_num_input, int64_t ffn_num_hiddens, int64_t num_heads,
		    double dropout, bool use_bias=false, torch::Device device=torch::kCPU) {

        attention = MultiHeadAttention(key_size, query_size, value_size, num_hiddens,
        		num_heads, dropout, use_bias, device);
        addnorm1 = AddNorm(norm_shape, dropout, device);
        ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens, device);
        addnorm2 = AddNorm(norm_shape, dropout, device);
	}

	torch::Tensor forward(torch::Tensor X, torch::Tensor valid_lens=torch::empty(0)) {
		torch::Tensor t = attention->forward(X, X, X, valid_lens);
		torch::Tensor Y = addnorm1->forward(X, t);
		torch::Tensor t2 = ffn->forward(Y);
		Y = addnorm2->forward(Y, t2);
        return Y;
    }

	PositionWiseFFN ffn{nullptr};
	AddNorm addnorm1{nullptr}, addnorm2{nullptr};
	MultiHeadAttention attention{nullptr};
};
TORCH_MODULE(TransformerEncoderBlock);

struct BERTEncoderImpl : public torch::nn::Module {
	//BERT编码器
	BERTEncoderImpl(int64_t vocab_size, int64_t num_hiddens, std::vector<int64_t> norm_shape, int64_t ffn_num_input,
    		   int64_t ffn_num_hiddens, int64_t num_heads, int64_t num_layers, double dropout,
			   int64_t max_len=1000, int64_t key_size=768, int64_t query_size=768,
			   int64_t value_size=768, torch::Device device=torch::kCPU) {

    	// int64_t vocab_size, int64_t num_hiddens, int64_t  ffn_num_hiddens,
	    // int64_t  num_heads, int64_t  num_blks, double dropout, int64_t  max_len=1000) {

        token_embedding = torch::nn::Embedding(vocab_size, num_hiddens);
        token_embedding->to(device);
        segment_embedding = torch::nn::Embedding(2, num_hiddens);
        segment_embedding->to(device);
        blks = torch::nn::Sequential();

        for(int64_t  i = 0; i < num_layers; i++ ) {
        	blks->push_back(TransformerEncoderBlock(key_size, query_size, value_size, num_hiddens, norm_shape,
                ffn_num_input, ffn_num_hiddens, num_heads, dropout, true, device));
        }
        //In BERT, positional embeddings are learnable, thus we create a
        //parameter of positional embeddings that are long enough
        //pos_embedding = torch::nn::Parameter(torch::randn({1, max_len,
        //                                              num_hiddens}));
        pos_embedding = torch::randn({1, max_len, num_hiddens}).to(device);
	}

    torch::Tensor forward(torch::Tensor tokens, torch::Tensor segments, torch::Tensor valid_lens=torch::empty(0)) {
        // Shape of `X` remains unchanged in the following code snippet:
        // (batch size, max sequence length, `num_hiddens`)
    	torch::Tensor X = token_embedding(tokens) + segment_embedding(segments);
        X = X + pos_embedding.index({Slice(), Slice(None, X.size(1)), Slice()}); //[:, :X.shape[1], :]

        for(auto& blk : *blks.ptr()) {
            X = blk.forward(X, valid_lens);
        }
        return X;
    }

    torch::nn::Embedding token_embedding{nullptr}, segment_embedding{nullptr};
    torch::nn::Sequential blks;
    torch::Tensor pos_embedding;
};
TORCH_MODULE(BERTEncoder);

struct MaskLMImpl : public torch::nn::Module {
    // BERT的掩蔽语言模型任务
	MaskLMImpl(int64_t vocab_size, int64_t num_hiddens, int64_t num_inputs=768, torch::Device device=torch::kCPU) {
		std::vector<int64_t> norm_shape;
		norm_shape.push_back(num_hiddens);
        mlp = torch::nn::Sequential(
        	  torch::nn::Linear(torch::nn::LinearOptions(num_inputs, num_hiddens).bias(true)),
              torch::nn::ReLU(),
			  torch::nn::LayerNorm(torch::nn::LayerNormOptions(norm_shape)),
			  torch::nn::Linear(torch::nn::LinearOptions(num_hiddens, vocab_size).bias(true))
		);
        mlp->to(device);
	}

	torch::Tensor forward(torch::Tensor X, torch::Tensor pred_positions) {
        auto num_pred_positions = pred_positions.size(1);
        pred_positions = pred_positions.reshape(-1);
        auto batch_size = X.size(0);
        auto batch_idx = torch::arange(0, batch_size).to(X.device());
        // 假设batch_size=2，num_pred_positions=3
        // 那么batch_idx是np.array（[0,0,0,1,1,1]）
        c10::optional<int64_t> dim = {0};
        batch_idx = torch::repeat_interleave(batch_idx, num_pred_positions, dim).to(X.device()); //, 0)
        auto masked_X = X.index({batch_idx, pred_positions});
        masked_X = masked_X.reshape({batch_size, num_pred_positions, -1});
        return mlp->forward(masked_X);
	}

	torch::nn::Sequential mlp{nullptr};
};
TORCH_MODULE(MaskLM);

struct NextSentencePredImpl : public torch::nn::Module {
    //BERT的下一句预测任务
	NextSentencePredImpl(int64_t num_inputs, torch::Device device=torch::kCPU) {
        //self.output = nn.Dense(num_inputs, 2)
		output = torch::nn::Linear(torch::nn::LinearOptions(num_inputs, 2).bias(true));
		output->to(device);
	}

    torch::Tensor forward(torch::Tensor X) {
        // X的形状：(batchsize,num_hiddens)
        return output->forward(X);
	}
    torch::nn::Linear output{nullptr};
};
TORCH_MODULE(NextSentencePred);

// 整合代码
struct BERTModelImpl : public torch::nn::Module {
    //BERT模型
	BERTModelImpl(int64_t vocab_size, int64_t num_hiddens, std::vector<int64_t> norm_shape, int64_t ffn_num_input,
			int64_t ffn_num_hiddens, int64_t num_heads, int64_t num_layers, double dropout, int64_t max_len=1000,
			int64_t key_size=768, int64_t query_size=768, int64_t value_size=768, int64_t hid_in_features=768,
			int64_t mlm_in_features=768, int64_t nsp_in_features=768, torch::Device device=torch::kCPU) {
        encoder = BERTEncoder(vocab_size, num_hiddens, norm_shape,
                    ffn_num_input, ffn_num_hiddens, num_heads, num_layers,
                    dropout, max_len, key_size, query_size, value_size, device);
        hidden = torch::nn::Sequential(
        		 torch::nn::Linear(torch::nn::LinearOptions(hid_in_features, num_hiddens)),
                 torch::nn::Tanh());
        hidden->to(device);
        mlm = MaskLM(vocab_size, num_hiddens, mlm_in_features, device);
        nsp = NextSentencePred(nsp_in_features, device);
	}

	std::tuple <torch::Tensor, torch::Tensor, torch::Tensor> forward(torch::Tensor tokens,
			torch::Tensor segments, torch::Tensor valid_lens=torch::empty(0),
			torch::Tensor pred_positions=torch::empty(0)) {

        torch::Tensor encoded_X = encoder->forward(tokens, segments, valid_lens);

        torch::Tensor mlm_Y_hat;
        if( pred_positions.numel() != 0 ) {
            mlm_Y_hat = mlm->forward(encoded_X, pred_positions).to(tokens.device());
        } else {
            mlm_Y_hat = torch::empty(0).to(tokens.device());
        }

        // 用于下一句预测的多层感知机分类器的隐藏层，0是“<cls>”标记的索引
        // nsp_Y_hat = self.nsp(self.hidden(encoded_X[:, 0, :]))
        torch::Tensor nsp_Y_hat = nsp->forward(hidden->forward(encoded_X.index({Slice(), 0, Slice()})));
        return std::make_tuple(encoded_X, mlm_Y_hat, nsp_Y_hat);
    }

	BERTEncoder encoder{nullptr};
	torch::nn::Sequential hidden{nullptr};
	MaskLM mlm{nullptr};
	NextSentencePred nsp{nullptr};
};
TORCH_MODULE(BERTModel);


#endif /* SRC_UTILS_CH_14_UTIL_H_ */
