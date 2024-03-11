#include <torch/torch.h>
#include <torch/script.h>
#include <torch/autograd.h>
#include <torch/utils.h>
#include <iostream>
#include <unistd.h>
#include <iomanip>
#include "../csvloader.h"

int main() {

	std::cout << "Current path is " << get_current_dir_name() << '\n';

	// Load CSV data
	std::ifstream file;
	std::string path = "./data/house_tiny.csv";
	file.open(path, std::ios_base::in);

	// Exit if file not opened successfully
	if(!file.is_open()) {
		std::cout << "File not read successfully" << std::endl;
		std::cout << "Path given: " << path << std::endl;
		return -1;
	}

	int num_records = std::count(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), '\n');
	std::cout << "records in file = " << num_records << '\n';

	// ste file read from begining
	file.clear();
	file.seekg(0, std::ios::beg);

	CSVRow row;
	// Read and throw away the first row.
	// file >> row

	while (file >> row) {
		std::vector<std::string> r_data = row.getRowData();
		for (std::size_t loop = 0; loop < r_data.size(); ++loop) {
			if( isNumber( r_data[loop].c_str() ) ) {
				std::cout << std::atof(r_data[loop].c_str()) << " ";
			} else {
				std::cout << r_data[loop].c_str() << " ";
			}
		}
		std::cout << std::endl;
	}
	file.close();

	std::cout << "Done!\n";
	return 0;
}




