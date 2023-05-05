#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include <string>
#include <valarray>
#include <cassert>

#include "../utils.h"
#include "../csvloader.h"

#include <matplot/matplot.h>
using namespace matplot;

using torch::indexing::Slice;
using torch::indexing::None;


// This function processes data, Loads CSV file to vectors and normalizes features to (0, 1)
// Assumes last column to be label and first row to be header (or name of the features)
std::vector<std::vector<float>> get_feature_data(std::ifstream& file, size_t train_size) {
	std::vector<std::vector<float>> features;
	std::vector<float> label;

	CSVRow  row;
    // Read and throw away the first row.
    file >> row;

    size_t t = 0;
	while (file >> row && t < train_size) {
		std::vector<float> rdata;
		for (std::size_t loop = 0;loop < row.size(); ++loop) {
			rdata.push_back(row[loop]);
		}
		features.push_back(rdata);
		t++;
	}
	file.close();

//	torch::Tensor data = torch::zeros({static_cast<int64_t>(features.size()), static_cast<int64_t>(features[0].size())},
//			torch::TensorOptions().dtype(torch::kFloat));

//	for( int i = 0; i < features.size(); i++ ) {
//		std::vector<float> tmp = features[i];
//		for( int j = 0; j < tmp.size(); j++ ) {
//			data.index({i,j}) = tmp[j];
//		}
//	}

	return features;
}

std::vector<float> get_label_data(std::ifstream& file) {
	std::vector<float> label;
	std::string row;
    // Read and throw away the first row.
    file >> row;

	while (file >> row) {
		label.push_back(std::stof(row));
	}
	file.close();

//	torch::Tensor tlabels = torch::zeros(static_cast<int64_t>(label.size()),
//			torch::TensorOptions().dtype(torch::kFloat));

//	for( int i = 0; i < label.size(); i++ ) {
//		tlabels.index({i}) = label[i];
//	}

	return label;
}

void get_k_fold_data(int64_t k, int64_t i, std::vector<std::vector<float>> X, std::vector<float> y,
					 torch::Tensor& X_train, torch::Tensor& y_train,
					 torch::Tensor& X_valid, torch::Tensor& y_valid) {
    assert(k > 1);

    std::valarray<int64_t> index(y.size());
    for (int k=0; k<12; ++k) index[k]=k;

    int64_t fold_size = static_cast<int64_t>(X.size() *1.0 / k);
    //X_train, y_train = None, None
    for(int64_t j = 0; j < k; j++ ) {
    	//std::slice idx = std::slice(j * fold_size, (j + 1) * fold_size, 1);
    	std::valarray<int64_t> X_index = index[std::slice(j * fold_size, fold_size, 1)]; // (j + 1) *
        //auto y_index = index[std::slice(j * fold_size, (j + 1) * fold_size, 1)];

        if( j == i ) {
        	y_valid = torch::zeros(static_cast<int64_t>(X_index.size()),
        				torch::TensorOptions().dtype(torch::kFloat));

            X_valid = torch::zeros({static_cast<int64_t>(X_index.size()), static_cast<int64_t>(X[0].size())},
        			torch::TensorOptions().dtype(torch::kFloat));

            for( int r = 0; r < X_index.size(); r++ ) {
            	std::vector<float> tmp = X[X_index[r]];
            	for( int c = 0; c < tmp.size(); c++ ) {
            		X_valid.index({r,c}) = tmp[c];
            	}
            	y_valid.index({r}) = y[X_index[r]];
            }

        } else if( X_train.size(0) == 0 ) {
            //X_train, y_train = X_part, y_part
        	y_train = torch::zeros(static_cast<int64_t>(X_index.size()),
        	        				torch::TensorOptions().dtype(torch::kFloat));

        	X_train = torch::zeros({static_cast<int64_t>(X_index.size()), static_cast<int64_t>(X[0].size())},
        	        			torch::TensorOptions().dtype(torch::kFloat));

        	for( int r = 0; r < X_index.size(); r++ ) {
        	    std::vector<float> tmp = X[X_index[r]];
        	    for( int c = 0; c < tmp.size(); c++ ) {
        	        X_train.index({r,c}) = tmp[c];
        	    }
        	    y_train.index({r}) = y[X_index[r]];
        	}

        }else {
        	auto y_tmp = torch::zeros(static_cast<int64_t>(X_index.size()),
        	        				torch::TensorOptions().dtype(torch::kFloat));

        	auto X_tmp= torch::zeros({static_cast<int64_t>(X_index.size()), static_cast<int64_t>(X[0].size())},
        	        			torch::TensorOptions().dtype(torch::kFloat));

        	for( int r = 0; r < X_index.size(); r++ ) {
        	    std::vector<float> tmp = X[X_index[r]];
        	    for( int c = 0; c < tmp.size(); c++ ) {
        	        X_tmp.index({r,c}) = tmp[c];
        	    }
        	    y_tmp.index({r}) = y[X_index[r]];
        	}

            X_train = torch::cat({X_train, X_tmp}, 0);
            y_train = torch::cat({y_train, y_tmp}, 0);
        }
    }
}

float log_rmse(torch::nn::Sequential net, torch::nn::MSELoss loss, torch::Tensor features, torch::Tensor labels) {
	//To further stabilize the value when the logarithm is taken, set the
	// value less than 1 as 1
	auto clipped_preds = torch::clamp(net->forward(features), 1);
	auto rmse = torch::sqrt(loss(torch::log(clipped_preds), torch::log(labels)));
	return rmse.item<float>();
}


int main( void ) {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Load CSV data
	std::ifstream file;
	std::string path = "./data/kaggle_house_data.csv";
	file.open(path, std::ios_base::in);
	// Exit if file not opened successfully
	if (!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	std::ifstream file2;
	std::string path2 = "./data/kaggle_house_train_data_labels.csv";
	file2.open(path2, std::ios_base::in);
	// Exit if file not opened successfully
	if (!file2.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path2 << std::endl;
		return -1;
	}

	// Process Data, load features and labels for LR
	std::vector<float> train_labels = get_label_data(file2);
	std::cout << train_labels.size() << std::endl;

	std::vector<std::vector<float>> features = get_feature_data(file, train_labels.size());
	//std::cout << features.index({Slice(None, 10), Slice(None, 10)}) << std::endl;
	std::cout << features.size() << std::endl;

	int64_t n_train = train_labels.size();

	//auto train_features = features.index({Slice(None, n_train), Slice()});
	//auto test_features = features.index({Slice(n_train, None), Slice()});

	// Training
	auto loss = torch::nn::MSELoss();
	int64_t in_features = features[0].size();

	int64_t k=5, num_epochs=100, batch_size=64;
	float train_l_sum=0, valid_l_sum=0, lr=5, weight_decay=0;

    std::vector<float> train_ls_epoch (num_epochs);
    std::vector<float> test_ls_epoch (num_epochs);
    std::vector<float> xx (num_epochs);

    for( int64_t epoch=0; epoch < num_epochs; epoch++ ) {
    	train_ls_epoch[epoch] = 0;
    	test_ls_epoch[epoch] = 0;
    }

	for(int64_t i = 0; i < k; i++ ) {
		torch::Tensor X_train, y_train, X_valid, y_valid;

		get_k_fold_data( k, i, features, train_labels, X_train, y_train, X_valid, y_valid);
		//std::cout << X_train.sizes() << " " << y_train.sizes() << std::endl;
		//std::cout << X_valid.sizes() << " " << y_valid.sizes() << std::endl;

	    // model
	    auto net = torch::nn::Sequential(torch::nn::Linear(in_features, 1));

	    float train_ls=0, test_ls=0;
	    // train_iter = d2l.load_array((train_features, train_labels), batch_size)
	    auto train_data = LRdataset(X_train, y_train)
	    						   .map(torch::data::transforms::Stack<>());
	    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
	    		                   std::move(train_data), batch_size);

	    // The Adam optimization algorithm is used here
	    auto optimizer = torch::optim::Adam(net->parameters(), torch::optim::AdamOptions(lr).weight_decay(weight_decay));

	    for( int64_t epoch=0; epoch < num_epochs; epoch++ ) {
	    	for(auto& batch : *data_loader ) {
	            auto X = batch.data;
	            auto y = batch.target;

	    		optimizer.zero_grad();
	            auto l = loss(net->forward(X), y.reshape({-1,1}));
	            l.backward();
	            optimizer.step();
	    	}

	    	train_ls = log_rmse(net, loss, X_train, y_train.reshape({-1,1}));
	        if( y_valid.size(0) != 0 )
	            test_ls = log_rmse(net, loss, X_valid, y_valid.reshape({-1,1}));

	        //std::cout << train_ls << std::endl;

	        train_ls_epoch[epoch] += train_ls;
	        test_ls_epoch[epoch] += test_ls;

	        if( i == 0 ) xx[epoch] = (epoch + 1);
	    }
	    std::cout <<"fold " << (i + 1) << ", train log rmse:" << train_ls << ", valid log rmse: " << test_ls << std::endl;
	}

    for( int64_t epoch=0; epoch < num_epochs; epoch++ ) {
    	train_l_sum += train_ls_epoch[epoch];
    	train_ls_epoch[epoch] = (train_ls_epoch[epoch]/k);
    	valid_l_sum += test_ls_epoch[epoch];
    	test_ls_epoch[epoch] = (test_ls_epoch[epoch]/k);
    }
    train_l_sum /= num_epochs;
    valid_l_sum /= num_epochs;

	auto F = figure(true);
	F->size(800, 600);
	F->add_axes(false);
	F->reactive_mode(false);
	F->tiledlayout(1, 1);
	F->position(0, 0);

	auto ax1 = F->nexttile();
	matplot::hold(ax1, true);
	matplot::semilogy(ax1, xx, train_ls_epoch, "b")->line_width(2);
	matplot::semilogy(ax1, xx, test_ls_epoch, "c:")->line_width(2);
	matplot::hold(ax1, false);
	matplot::xlabel(ax1, "epoch");
	matplot::ylabel(ax1, "log(loss)");
	matplot::legend(ax1, {"Train loss", "Test loss"});
	matplot::show();

    std::cout << "5-fold validation: avg train log rmse:" << train_l_sum << ", avg valid log rmse: " << valid_l_sum << std::endl;

	std::cout << "Done!\n";
	return 0;
}


