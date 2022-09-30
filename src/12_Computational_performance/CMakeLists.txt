cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(12_AsynchronousComputation)
target_sources(12_AsynchronousComputation PRIVATE 
AsynchronousComputation.cpp 
../utils.h
../utils.cpp						
)
													
target_include_directories(12_AsynchronousComputation PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(12_AsynchronousComputation ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(12_AsynchronousComputation PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)


#----------------------------------------------------------------------------------
add_executable(12_Hybridize)
target_sources(12_Hybridize PRIVATE Hybridize.cpp )
													
target_include_directories(12_Hybridize PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(12_Hybridize ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(12_Hybridize PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)



