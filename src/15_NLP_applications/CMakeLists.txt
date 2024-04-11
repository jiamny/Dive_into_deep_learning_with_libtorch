
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

# ---------------------------------------------------------------
add_executable(15_SentimentAnalysisAndDataset)

target_sources(15_SentimentAnalysisAndDataset PRIVATE
SentimentAnalysisAndDataset.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_SentimentAnalysisAndDataset ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_SentimentAnalysisAndDataset PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(15_SentimentAnalysisRNN)

target_sources(15_SentimentAnalysisRNN PRIVATE
SentimentAnalysisRNN.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_SentimentAnalysisRNN ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_SentimentAnalysisRNN PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(15_SentimentAnalysisCNN)

target_sources(15_SentimentAnalysisCNN PRIVATE
SentimentAnalysisCNN.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_SentimentAnalysisCNN ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_SentimentAnalysisCNN PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(15_NaturalLanguageInferenceDataset)

target_sources(15_NaturalLanguageInferenceDataset PRIVATE
NaturalLanguageInferenceDataset.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_NaturalLanguageInferenceDataset ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_NaturalLanguageInferenceDataset PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(15_NaturalLanguageInferenceAttention)

target_sources(15_NaturalLanguageInferenceAttention PRIVATE
NaturalLanguageInferenceAttention.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_NaturalLanguageInferenceAttention ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_NaturalLanguageInferenceAttention PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(15_NaturalLanguageInferenceBert)

target_sources(15_NaturalLanguageInferenceBert PRIVATE
NaturalLanguageInferenceBert.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_14_util.h
../utils/ch_14_util.cpp
../utils/ch_15_util.h
../utils/ch_15_util.cpp
)

target_link_libraries(15_NaturalLanguageInferenceBert ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(15_NaturalLanguageInferenceBert PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)








