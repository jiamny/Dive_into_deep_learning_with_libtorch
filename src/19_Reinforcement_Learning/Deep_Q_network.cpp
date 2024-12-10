#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <random>
#include <cmath>
#include <bits/stdc++.h>
#include "../TempHelpFunctions.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

template <typename T>
bool IsEqual(T rhs, T lhs) {
    T diff = std::abs(lhs - rhs);

    T epsilon = std::numeric_limits<T>::epsilon() * std::max(std::abs(rhs), std::abs(lhs));

    return diff <= epsilon ;
}

class Env {
public:
	std::vector<std::string> action_space;
	int n_actions = 0, x1 = 0, y1 = 0, end_game = 0;
	torch::Tensor migong;
	Mat frame;

    Env() {
        action_space = {"u", "d", "l", "r"};
        n_actions = action_space.size();
        frame = Mat(300, 300, CV_8UC3, cv::Scalar(255, 255, 255));
        start_env();
    }

    torch::Tensor start_env(void) {
        migong = torch::tensor({{1, 0, 0, 0, 0},
                       {0, 0, 0, 3, 0},
                       {0, 0, 0, 0, 0},
                       {0, 3, 0, 0, 0},
                       {0, 0, 0, 0, 2}});
        x1 = 0;
        y1 = 0;
        end_game = 0;
        return migong;
    }

    void display(void) {
        Mat frameFace = frame.clone();

        for(auto& i : range(5, 0) )  {
            cv::line(frameFace, {i * 60, 0}, {i * 60, 300}, {0, 0, 0}, 1);
            cv::line(frameFace, {0, i * 60}, {300, i * 60}, {0, 0, 0}, 1);
        }

        for(auto& x : range(5, 0)) {
            for(auto& y : range(5, 0)) {
                if(migong[y][x].data().item<int>() == 1)
                    cv::circle(frameFace, {x * 60 + 30, y * 60 + 30}, 25, {255, 0, 0}, -1);
                if(migong[y][x].data().item<int>() == 2)
                    cv::circle(frameFace, {x * 60 + 30, y * 60 + 30}, 25, {0, 255, 0}, -1);
                if(migong[y][x].data().item<int>() == 3)
                    cv::circle(frameFace, {x * 60 + 30, y * 60 + 30}, 25, {0, 0, 255}, -1);
            }
        }

        cv::imshow(" ", frameFace);
        cv::waitKey(10);
    }

    std::tuple<int, torch::Tensor, torch::Tensor> step(int action) {
        float r = 0.;
        // ['u'0, 'd'1, 'l'2, 'r'3]
        if(action == 0) {
            if( y1 == 0 ) {
                r = -0.5;
            } else {
                migong[y1][x1] = 0;
                migong[y1 - 1][x1] = 1;
                y1 -= 1;
                if( y1 == 1 && x1 == 3 ) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1.;
                } else if( y1 == 4 && x1 == 4 ) {
                    end_game = 2;
                    r = 1.;
                }
            }
        }
        if( action == 1 ) {
            if(y1 == 4) {
                r = -0.5;
            } else {
                migong[y1][x1] = 0;
                migong[y1 + 1][x1] = 1;
                y1 += 1;
                if(y1 == 1 && x1 == 3 ) {
                    end_game = 1;
                    r = -1.;
                } else if( y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1.;
                }
            }
        }

        if(action == 2) {
            if(x1 == 0) {
                r = -0.5;
            } else {
                migong[y1][x1] = 0;
                migong[y1][x1 - 1] = 1;
                x1 -= 1;
                if( y1 == 1 && x1 == 3 ) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1.;
                }
            }
        }

        if( action == 3 ) {
            if(x1 == 4) {
                r = -0.5;
            } else {
                migong[y1][x1] = 0;
                migong[y1][x1 + 1] = 1;
                x1 += 1;
                if(y1 == 1 && x1 == 3) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1.;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1.;
                }
            }
        }

        return std::make_tuple(end_game, torch::tensor({r}), migong);
    }
};

struct NetImpl : public torch::nn::Module {
public:
	torch::nn::Conv2d c1{nullptr};
	torch::nn::Linear f1{nullptr}, f2{nullptr};
    NetImpl(){
        c1=torch::nn::Conv2d(torch::nn::Conv2dOptions(3, 25, 5).stride(1).padding(0));
        f1=torch::nn::Linear(torch::nn::LinearOptions(25,16));
        f1->weight.data().normal_(0., 0.1);
        f2=torch::nn::Linear(torch::nn::LinearOptions(16,4));
        f2->weight.data().normal_(0., 0.1);
        register_module("c1", c1);
        register_module("f1", f1);
        register_module("f2", f2);

    }

    torch::Tensor forward(torch::Tensor x) {
        x = c1->forward(x);
        x = torch::nn::functional::relu(x);
        x = x.view({x.size(0),-1});
        x = f1->forward(x);
        x = torch::nn::functional::relu(x);
		torch::Tensor action= f2->forward(x);
        return action;
    }
};
TORCH_MODULE(Net);

struct DQN {
    Net eval_net{nullptr},  target_net{nullptr};
    int MEMORY_CAPACITY = 2000, memory_counter, learn_step_counter,
    	N_ACTIONS = 4, TARGET_REPLACE_ITER = 100, BATCH_SIZE = 8;
    std::vector<std::vector<torch::Tensor>> memory;
    //torch::optim::Optimizer optimizer;
    torch::nn::MSELoss loss_func{nullptr};
    float LR = 0.01, EPSILON = 0.9, GAMMA = 0.9;

    DQN(){
        eval_net = Net();
        target_net = Net();

        learn_step_counter = 0;
        memory_counter = 0;
        //memory = torch::zeros({MEMORY_CAPACITY, 4});
        for(auto& _ : range(MEMORY_CAPACITY, 0))  {
        	std::vector<torch::Tensor> t;
        	for(auto& i : range(4, 0))
        		t.push_back( torch::empty(0) );
        	memory.push_back(t);
        }

        loss_func = torch::nn::MSELoss();
    }

    torch::Tensor choose_action(torch::Tensor x) {
        x = torch::unsqueeze(x, 0).to(torch::kFloat32);
        torch::Tensor action, _;
        // select optimized action
        if( torch::rand({1}).data().item<float>() < EPSILON )  {
            torch::Tensor actions_value = eval_net->forward(x);
            std::tie(_, action) = torch::max(actions_value, 1);
        } else {   // select random action
            action = torch::randint(0, N_ACTIONS, {1});
        }
        return action;
    }

    void store_transition(torch::Tensor s, torch::Tensor a, torch::Tensor r, torch::Tensor s_) {
        //if memory full
        int index = memory_counter % MEMORY_CAPACITY;
        memory[index][0] = s;
        memory[index][1] = a;
        memory[index][2] = r;
        memory[index][3] = s_;
        memory_counter += 1;
    }

    void learn(torch::optim::Optimizer& optimizer) {
        // update target net parameters per 100 runs
        if( learn_step_counter % TARGET_REPLACE_ITER == 0) {
            // copy eval_net parameters to target_net
            //target_net->load_state_dict(eval_net->state_dict());
        	 torch::autograd::GradMode::set_enabled(false);  // make parameters copying possible
        	 auto new_params = eval_net->named_parameters(true); //
        	 auto new_buffers = eval_net->named_buffers(true);
        	 auto params = target_net->named_parameters(true /*recurse*/);
        	 auto buffers = target_net->named_buffers(true /*recurse*/);
        	 for(auto& val : new_params) {
        	     auto name = val.key();
        	     auto* t = params.find(name);
        	     if (t != nullptr) {
        	          t->copy_(val.value());
        	      }
        	 }
        	 for(auto& val : new_buffers) {
        	     auto name = val.key();
        	     auto* t = buffers.find(name);
        	     if (t != nullptr) {
        	          t->copy_(val.value());
        	      }
        	}
        	torch::autograd::GradMode::set_enabled(true);
        }
        learn_step_counter += 1;

        // random select BATCH_SIZE data index in range of MEMORY_CAPACITY
        torch::Tensor sample_index = torch::randint(0, MEMORY_CAPACITY, {BATCH_SIZE});
        //sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        std::vector<torch::Tensor> tb_s, tb_a, tb_r, tb_s_;
        for(auto& i : range(BATCH_SIZE, 0) ) {
        	int idx = sample_index[i].data().item<int>();
            tb_s.push_back( memory[idx][0] );
            tb_a.push_back( memory[idx][1] );
            tb_r.push_back( memory[idx][2] );
            tb_s_.push_back( memory[idx][3] );
        }

        torch::Tensor b_s = torch::stack(tb_s, 0).to(torch::kFloat32);
        torch::Tensor b_a = torch::stack(tb_a, 0).to(torch::kInt64);
        torch::Tensor b_r = torch::stack(tb_r, 0).to(torch::kFloat32);
        torch::Tensor b_s_ = torch::stack(tb_s_, 0).to(torch::kFloat32);

        // based on b_a, select q_eval value
        torch::Tensor q_eval = eval_net->forward(b_s).gather(1, b_a);  // shape (batch, 1) find Q value of action
        torch::Tensor q_next = target_net->forward(b_s_).detach();     // q_next no need backward, detach it

        torch::Tensor max_q, _;
        std::tie(max_q, _) = q_next.max(1);
        torch::Tensor q_target = b_r + GAMMA * max_q.reshape({-1, 1});

        auto loss = loss_func(q_eval, q_target);

        // update eval net
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }
};

torch::Tensor trans_torch(torch::Tensor s) {
	torch::Tensor l1 = torch::where(s==1,1,0);
	torch::Tensor l2 = torch::where(s==2,1,0);
	torch::Tensor l3 = torch::where(s==3,1,0);
	torch::Tensor b = torch::stack({l1,l2,l3}, 0);
    return b;
}

int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	Env env = Env();
	DQN dqn = DQN();
	auto optimizer = torch::optim::Adam(dqn.eval_net->parameters(), dqn.LR);

	int study = 1;
	int Failed = 0;
	int Succeeded = 0;
	int num_games = 200;
	for(auto& i_episode : range(num_games, 0) ) {

	    torch::Tensor s = env.start_env();
	    s = trans_torch(s);

	    while(true) {
	        env.display();   							//show animation

	        torch::Tensor a = dqn.choose_action(s); 	//select action

	        //action, receive env's reward
	        int done;
	        torch::Tensor r, s_;
	        std::tie(done, r, s_) = env.step(a.data().item<int>());
	        s_ = trans_torch(s_);

	        //update memory
	        dqn.store_transition(s, a, r, s_);

	        if( dqn.memory_counter > dqn.MEMORY_CAPACITY) {
	            if( study == 1 )
	                study = 0;
	            dqn.learn(optimizer);	// start learning after memory full
	        }

	        // when ends at end, start new game
	        if( done == 1 || done == 2 ) {
	            if(done == 1) {
	                std::cout << "Epoch: " << (i_episode+1) << " reward: " << r.data().item<float>() << " Failed\n";
	                Failed += 1;
	            }
	            if(done == 2) {
	            	std::cout << "Epoch: " << (i_episode+1) << " reward: " << r.data().item<float>() << " Succeeded\n";
	                Succeeded += 1;
	            }
	            break;
	        }
	        s = s_.clone();
	    }

	}
	printf("Number of play game = %d, Succeeded = %d, Failed = %d\n", num_games, Succeeded, Failed);

	std::cout << "Done!\n";
}





