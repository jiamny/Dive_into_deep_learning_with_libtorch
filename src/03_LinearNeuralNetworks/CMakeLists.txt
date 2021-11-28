cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------

add_executable(03_linear_regression)

target_sources(03_linear_regression PRIVATE linear_regression.cpp 
													../utils.h 
													../utils.cpp
													)
													
target_include_directories(03_linear_regression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_linear_regression ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_linear_regression PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(03_linear_regression_concise)

target_sources(03_linear_regression_concise PRIVATE linear_regression_concise.cpp 
													../utils.h 
													../utils.cpp
													)
													
target_link_libraries(03_linear_regression_concise ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)
											
set_target_properties(03_linear_regression_concise PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(03_linear_regression_scratch)

target_sources(03_linear_regression_scratch PRIVATE linear_regression_scratch.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_include_directories(03_linear_regression_scratch PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_linear_regression_scratch ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_linear_regression_scratch PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(03_load_image_dataset)

target_sources(03_load_image_dataset PRIVATE load_image_dataset.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_include_directories(03_load_image_dataset PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_load_image_dataset ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_load_image_dataset PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(03_softmax_regression_concise)

target_sources(03_softmax_regression_concise PRIVATE softmax_regression_concise.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_include_directories(03_softmax_regression_concise PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_softmax_regression_concise ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_softmax_regression_concise PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(03_softmax_regression_scratch)

target_sources(03_softmax_regression_scratch PRIVATE softmax_regression_scratch.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_include_directories(03_softmax_regression_scratch PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(03_softmax_regression_scratch ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(03_softmax_regression_scratch PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)



