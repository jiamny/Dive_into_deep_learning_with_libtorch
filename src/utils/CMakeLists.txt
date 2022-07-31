cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.9")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

#----------------------------------------------------------------------------------
add_executable(Utils_img_loder_demo)

target_sources(Utils_img_loder_demo PRIVATE img_loder_demo.cpp 
						../utils.h 
						../utils.cpp
						ch_13_util.h
						ch_13_util.cpp
						dataloader.hpp
						dataloader.cpp
						datasets.hpp
						datasets.cpp
						transforms.hpp
						transforms.cpp
						)
													
target_include_directories(Utils_img_loder_demo PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(Utils_img_loder_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(Utils_img_loder_demo PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------------
add_executable(Utils_matplotlib_subplot)

target_sources(Utils_matplotlib_subplot PRIVATE Matplotlib_subplot.cpp)
													
target_include_directories(Utils_matplotlib_subplot PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(Utils_matplotlib_subplot ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(Utils_matplotlib_subplot PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)


