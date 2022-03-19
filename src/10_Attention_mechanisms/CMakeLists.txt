cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(10_Attention_cues)
target_sources(10_Attention_cues PRIVATE 
Attention_cues.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_Attention_cues PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_Attention_cues ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_Attention_cues PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_NadarayaWatsonKernelRegression)
target_sources(10_NadarayaWatsonKernelRegression PRIVATE 
NadarayaWatsonKernelRegression.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_NadarayaWatsonKernelRegression PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_NadarayaWatsonKernelRegression ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_NadarayaWatsonKernelRegression PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_AttentionScoringFunctions)
target_sources(10_AttentionScoringFunctions PRIVATE 
AttentionScoringFunctions.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp					
)
													
target_include_directories(10_AttentionScoringFunctions PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_AttentionScoringFunctions ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_AttentionScoringFunctions PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_MultiHeadAttention)
target_sources(10_MultiHeadAttention PRIVATE 
MultiHeadAttention.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_MultiHeadAttention PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_MultiHeadAttention ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_MultiHeadAttention PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_BahdanauAttention)
target_sources(10_BahdanauAttention PRIVATE 
BahdanauAttention.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_BahdanauAttention PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_BahdanauAttention ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_BahdanauAttention PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_SelfAttentionAndPositionalEncoding)
target_sources(10_SelfAttentionAndPositionalEncoding PRIVATE 
SelfAttentionAndPositionalEncoding.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_SelfAttentionAndPositionalEncoding PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_SelfAttentionAndPositionalEncoding ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_SelfAttentionAndPositionalEncoding PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_Transformer)
target_sources(10_Transformer PRIVATE 
Transformer.cpp 
../matplotlibcpp.h
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_include_directories(10_Transformer PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(10_Transformer ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(10_Transformer PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

