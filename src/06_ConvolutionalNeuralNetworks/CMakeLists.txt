cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

#----------------------------------------------------------------------------------

add_executable(06_Channels)

target_sources(06_Channels PRIVATE Channels.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(06_Channels PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(06_Channels ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(06_Channels PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

