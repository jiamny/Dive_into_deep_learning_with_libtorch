
#ifndef SRC_UTILS_TEMPHELPFUNCTIONS_HPP_
#define SRC_UTILS_TEMPHELPFUNCTIONS_HPP_
#pragma once

#include <algorithm>
#include <vector>
#include <string>
#include <iostream>

/*
 * There is no way to change the precision via to_string() but the setprecision IO manipulator could be used instead:
 */
template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 6)
{
    std::ostringstream out;
    out.precision(n);
    out << std::fixed << a_value;
    return out.str();
}


template<typename T>
std::vector<T> range(const T count, T start = 1) {
    std::vector<T> aVector(count);
    iota(aVector.begin(), aVector.end(), start);
    return aVector;
}

template<typename T>
void printVector(std::vector<T> data) {

	std::cout << "[ ";
    #if __cplusplus==201103L
		std::copy(data.begin(), data.end(), std::ostream_iterator<int>(std::cout, " "));
    #else
		std::for_each(data.begin(), data.end(), [](T v) {std::cout << v << " ";});
	#endif
	std::cout << "]\n";
}

template<typename T>
std::vector<T> truncate_pad(std::vector<T> line, size_t num_steps, T padding_token) {
    //Truncate or pad sequences."""
    if( line.size() > num_steps ) {
    	std::vector<T> tokens(&line[0], &line[num_steps]);
        return tokens;  // Truncate
    } else {
    	int num_pad = num_steps - line.size();
    	for( int i = 0; i < num_pad; i++ )
    		line.push_back(padding_token);
    	return line;
    }
}

template<typename T>
T vector_sum(std::vector<T> data) {
    T total = 0;
    if( data.size() > 0 ) {
    	total = std::accumulate(data.begin(), data.end(), 0);
    }
    return total;
}


#endif /* SRC_UTILS_TEMPHELPFUNCTIONS_HPP_ */
