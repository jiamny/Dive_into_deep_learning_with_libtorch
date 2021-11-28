
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <iostream>
#include <vector>


template <typename T>
void pretty_print(const std::string& info, T&& data) {
  std::cout << info << std::endl;
  std::cout << data << std::endl << std::endl;
}

void rnn_eample_1(void) {
  // Use GPU when present, CPU otherwise.
  torch::Device device(torch::kCPU);
  if (torch::cuda::is_available()) {
    device = torch::Device(torch::kCUDA);
    std::cout << "CUDA is available! Training on GPU." << std::endl;
  }

  const size_t kSequenceLen = 1;
  const size_t kInputDim = 1;
  const size_t kHiddenDim = 5;
  const size_t kOuputDim = 1;
  auto time_serie_detector = torch::nn::LSTM(torch::nn::LSTMOptions(kInputDim, kHiddenDim)
                                                 .dropout(0.2)
                                                 .num_layers(kSequenceLen)
                                                 .bidirectional(false));
  time_serie_detector->to(device);
  std::cout << time_serie_detector << std::endl;

  torch::Tensor input = torch::empty({kSequenceLen, kInputDim});
  torch::Tensor state = torch::zeros({2, kSequenceLen, kHiddenDim});
  auto input_acc = input.accessor<float, 2>();
  size_t count = 0;
  for (float i = 0.1; i < 0.4; i += 0.1) {
    input_acc[count][0] = i;
    count++;
  }
  input = input.toBackend(c10::Backend::CPU);
  state = state.toBackend(c10::Backend::CPU);
  std::cout << "input = " << input << std::endl;
  time_serie_detector->zero_grad();

  auto i_tmp = input.view({input.size(0), 1, -1});
  auto h_tmp = state.view({2, state.size(0) / 2, 1, -1});
  auto c_tmp = h_tmp.clone();

  pretty_print("input: ", i_tmp);
  pretty_print("state: ", h_tmp);

 // torch::Tensor output;
 // std::tuple<torch::Tensor, torch::Tensor> state_new;
  auto tt = time_serie_detector->forward(i_tmp);
  auto output = std::get<0>(tt);
 // auto state_new = std::get<1>(tt);
 // std::forward_as_tuple(output, std::tie(h_tmp, c_tmp)) = time_serie_detector->forward(i_tmp, std::make_tuple(h_tmp, c_tmp));
  pretty_print("rnn_output/output: ", output);
  //pretty_print("rnn_output/state: ", state_new);
}


#define INPUTS 1
#define SEQUENCE 3
#define BATCH 1
#define LAYERS 3
#define HIDDEN 2
#define DIRECTIONS 2
#define OUTPUTS 1

struct BLSTM_Model : torch::nn::Module
{
	torch::nn::LSTM lstm{ nullptr };
	torch::nn::LSTM reverse_lstm{ nullptr };
	torch::nn::Linear linear{ nullptr };

	BLSTM_Model(uint64_t layers, uint64_t hidden, uint64_t inputs)
	{
		lstm = register_module("lstm", torch::nn::LSTM(
                   torch::nn::LSTMOptions(inputs, hidden).num_layers(layers)));
		reverse_lstm = register_module("rlstm", torch::nn::LSTM(
                   torch::nn::LSTMOptions(inputs, hidden).num_layers(layers)));
		linear = register_module("linear", torch::nn::Linear(
                   hidden*DIRECTIONS, OUTPUTS));
	}

	torch::Tensor forward(torch::Tensor x)
	{
		//Reverse and feed into LSTM + Reversed LSTM
		auto lstm1 = lstm->forward(x.view({ x.size(0), BATCH, -1 }));
                   //[SEQUENCE,BATCH,FEATURE]
		auto lstm2 = reverse_lstm->forward(torch::flip(x, 0).view({ x.size(0), BATCH, -1 }));
		//Reverse Output from Reversed LSTM + Combine Outputs into one Tensor
		auto cat = torch::empty({DIRECTIONS, BATCH, x.size(0), HIDDEN});
                   //[DIRECTIONS,BATCH,SEQUENCE,FEATURE]
		cat[0] = std::get<0>(lstm1).view({ BATCH, x.size(0), HIDDEN });
		cat[1] = torch::flip(std::get<0>(lstm2).view({ BATCH, x.size(0), HIDDEN }), 1);
		//Feed into Linear Layer
		auto out = torch::sigmoid(linear->forward(
                  cat.view({BATCH, x.size(0), HIDDEN*DIRECTIONS})));
                   //[BATCH,SEQUENCE,FEATURE]
		return out;
	}
};

int main()
{
	//Input: 0.1, 0.2, 0.3 -> Expected Output: 0.4, 0.5, 0.6
	BLSTM_Model model = BLSTM_Model(LAYERS, HIDDEN, INPUTS);
	torch::optim::Adam optimizer(
          model.parameters(), torch::optim::AdamOptions(0.0001));
	//Input
	torch::Tensor input = torch::empty({ SEQUENCE, INPUTS });
	auto input_acc = input.accessor<float, 2>();
	size_t count = 0;
	for (float i = 0.1; i < 0.4; i+=0.1)
	{
		input_acc[count][0] = i;
		count++;
	}
	//Target
	torch::Tensor target = torch::empty({ SEQUENCE, OUTPUTS });
	auto target_acc = target.accessor<float, 2>();
	count = 0;
	for (float i = 0.4; i < 0.7; i+=0.1)
	{
		target_acc[count][0] = i;
		count++;
	}
	//Train
	for (size_t i = 0; i < 6000; i++)
	{
		torch::Tensor output = model.forward(input);
		auto loss = torch::mse_loss(output.view({SEQUENCE, OUTPUTS}), target);
		std::cout << "Loss "<< i << " : " << loss.item<float>() << std::endl;
		loss.backward();
		optimizer.step();
	}
	//Test: Response should be about (0.4, 0.5, 0.6)
	torch::Tensor output = model.forward(input);
	std::cout << output << std::endl;
	return EXIT_SUCCESS;
}




