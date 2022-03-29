cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(11_Optimization_intro)

target_sources(11_Optimization_intro PRIVATE Optimization_intro.cpp )
													
target_include_directories(11_Optimization_intro PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(11_Optimization_intro ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(11_Optimization_intro PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_MinibatchStochasticGradientDescent)
target_sources(11_MinibatchStochasticGradientDescent PRIVATE 
MinibatchStochasticGradientDescent.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_include_directories(11_MinibatchStochasticGradientDescent PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(11_MinibatchStochasticGradientDescent ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(11_MinibatchStochasticGradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

