#include <unistd.h>
#include <iomanip>
#include <torch/utils.h>
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

    std::tuple<int, int> start_env(void) {
        migong = torch::tensor({{1, 0, 0, 0, 0},
                       {0, 0, 0, 3, 0},
                       {0, 0, 0, 0, 0},
                       {0, 3, 0, 0, 0},
                       {0, 0, 0, 0, 2}});
        x1 = 0;
        y1 = 0;
        end_game = 0;
        return std::make_tuple(x1, y1);
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
        cv::waitKey(30);
    }

    std::tuple<int, float, int, int> step(int action) {
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
                    r = -1;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1;
                } else if( y1 == 4 && x1 == 4 ) {
                    end_game = 2;
                    r = 1;
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
                    r = -1;
                } else if( y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1;
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
                    r = -1;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1;
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
                    r = -1;
                } else if(y1 == 3 && x1 == 1) {
                    end_game = 1;
                    r = -1;
                } else if(y1 == 4 && x1 == 4) {
                    end_game = 2;
                    r = 1;
                }
            }
        }

        return std::make_tuple( end_game, r, x1, y1);
    }
};



class QLearningAgent {
	std::vector<int> actions;
	float learning_rate, discount_factor, epsilon;
	std::map<std::string, std::vector<float>> q_table;
public:
	QLearningAgent(std::vector<int> _actions, torch::Tensor migong) {
        //['u', 'd', 'l', 'r'] <=> [0, 1, 2, 3]
        actions = _actions;
        learning_rate = 0.01;
        discount_factor = 0.9;
        epsilon = 0.3;

        if( ! q_table.empty() ) {
        	q_table.clear();
        }

        std::cout << migong.sizes() << '\n';
        int row = migong.size(0);
        int col = migong.size(1);
        for(int i = 0; i < row; i++) {
        	for(int j = 0; j < col; j++) {
        		std::string state = std::to_string(i) + "_" + std::to_string(j);
        		std::vector<float> v  = {0.0, 0.0, 0.0, 0.0};
        		q_table[state] = v;
        	}
        }
	}

    // sampling <s, a, r, s'>
    void learn(std::string state, int action, float reward, std::string next_state) {
        float current_q = q_table[state][action];
        // update Q table
        std::vector<float> v = q_table[next_state];
        float max_v = *std::max_element(v.begin(), v.end());
        float new_q = reward + discount_factor * max_v;
        q_table[state][action] += learning_rate * (new_q - current_q);
    }

    // select action from Q-table
    int get_action(std::string state) {
    	int action = -1;
        float rs = torch::rand({1}).data().item<float>();

        if( rs < epsilon ) {
            // random search action
        	int idx = torch::randint(0, actions.size(), {1}).data().item<int>();
            action = actions[idx];
        } else {
            // select from q table
        	std::vector<float> state_action = q_table[state];
            action = arg_max(state_action);
        }

        return action;
    }

    int arg_max(std::vector<float> state_action) {
    	std::vector<int> max_index_list;
        float max_value = state_action[0];
        for(int index = 0;  index < state_action.size(); index++) {
        	float value = state_action[index];
            if( value > max_value ) {
            	max_index_list.clear();
                max_index_list.push_back(index);
                max_value = value;
            } else {
            	if( IsEqual(value, max_value) ) {
            		max_index_list.push_back(index);
            	}
            }
        }
        int max_index = torch::randint(0, max_index_list.size(), {1}).data().item<int>();
        return max_index_list[max_index];
    }
};


int main() {
	std::cout << "Current path is " << get_current_dir_name() << '\n';
	torch::manual_seed(123);

	Env env = Env();
	QLearningAgent agent = QLearningAgent(range(env.n_actions, 0), env.migong);

	int Failed = 0;
	int Succeeded = 0;
	int num_games = 200;
	for(auto& episode : range(num_games, 0) ) {
		int x, y;
	    std::tie(x, y) = env.start_env();
	    std::string state = std::to_string(x) + "_" + std::to_string(y);
	    while(true) {
	        env.display();
	        // agent generate action
	        int action = agent.get_action(state);
	        int done;
	        float reward;
	        std::tie(done, reward, x, y) = env.step(action);
	        std::string next_state = std::to_string(x) + "_" + std::to_string(y);
	        // update Q-table
	        agent.learn(state, action, reward, next_state);

	        // when ends at end, start new game
	        if( done == 1 || done == 2 ) {
	            if(done == 1) {
	                std::cout << "Epoch: " << episode << " reward: " << reward << " Failed\n";
	                Failed += 1;
	            }
	            if(done == 2) {
	            	std::cout << "Epoch: " << episode << " reward: " << reward << " Succeeded\n";
	                Succeeded += 1;
	            }
	            break;
	        }
	        state = next_state;
	    }
	}

	printf("Number of play game = %d, Succeeded = %d, Failed = %d\n", num_games, Succeeded, Failed);

	std::cout << "Done!\n";
}




