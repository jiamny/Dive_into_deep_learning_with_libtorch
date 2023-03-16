
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(10_Attention_QKV)
target_sources(10_Attention_QKV PRIVATE 
Attention_QKV.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_link_libraries(10_Attention_QKV ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_Attention_QKV PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_AttentionPooling)
target_sources(10_AttentionPooling PRIVATE 
AttentionPooling.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_link_libraries(10_AttentionPooling ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_AttentionPooling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_NadarayaWatsonKernelRegression)
target_sources(10_NadarayaWatsonKernelRegression PRIVATE 
NadarayaWatsonKernelRegression.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)

target_link_libraries(10_NadarayaWatsonKernelRegression ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_NadarayaWatsonKernelRegression PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_AttentionScoringFunctions)
target_sources(10_AttentionScoringFunctions PRIVATE 
AttentionScoringFunctions.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp					
)

target_link_libraries(10_AttentionScoringFunctions ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_AttentionScoringFunctions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_MultiHeadAttention)
target_sources(10_MultiHeadAttention PRIVATE 
MultiHeadAttention.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)

target_link_libraries(10_MultiHeadAttention ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_MultiHeadAttention PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_BahdanauAttention)
target_sources(10_BahdanauAttention PRIVATE 
BahdanauAttention.cpp 
../utils.h
../utils.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp								
)

target_link_libraries(10_BahdanauAttention ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_BahdanauAttention PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_SelfAttentionAndPositionalEncoding)
target_sources(10_SelfAttentionAndPositionalEncoding PRIVATE 
SelfAttentionAndPositionalEncoding.cpp 
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_link_libraries(10_SelfAttentionAndPositionalEncoding ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_SelfAttentionAndPositionalEncoding PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_Transformer)
target_sources(10_Transformer PRIVATE 
Transformer.cpp 
../utils.h
../utils.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_link_libraries(10_Transformer ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_Transformer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(10_VisionTransformers)
target_sources(10_VisionTransformers PRIVATE 
VisionTransformers.cpp 
../utils.h
../utils.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils/ch_10_util.h
../utils/ch_10_util.cpp						
)
													
target_link_libraries(10_VisionTransformers ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(10_VisionTransformers PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


