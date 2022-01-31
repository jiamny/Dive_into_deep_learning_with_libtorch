cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------

add_executable(06_Channels)

target_sources(06_Channels PRIVATE Channels.cpp )

target_link_libraries(06_Channels ${TORCH_LIBRARIES} ) 

set_target_properties(06_Channels PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(06_ConvolutionalLayers)

target_sources(06_ConvolutionalLayers PRIVATE ConvolutionalLayers.cpp)
													
target_link_libraries(06_ConvolutionalLayers ${TORCH_LIBRARIES})

set_target_properties(06_ConvolutionalLayers PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(06_PaddingAndStride)

target_sources(06_PaddingAndStride PRIVATE PaddingAndStride.cpp)
													
target_link_libraries(06_PaddingAndStride ${TORCH_LIBRARIES})

set_target_properties(06_PaddingAndStride PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(06_Pooling)

target_sources(06_Pooling PRIVATE Pooling.cpp)
													
target_link_libraries(06_Pooling ${TORCH_LIBRARIES})

set_target_properties(06_Pooling PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(06_LeNet)

target_sources(06_LeNet PRIVATE LeNet.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(06_LeNet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(06_LeNet ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(06_LeNet PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
