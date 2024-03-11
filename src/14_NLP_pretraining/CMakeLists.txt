
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
add_executable(14_BERT_pretraining_dataset)

target_sources(14_BERT_pretraining_dataset PRIVATE
BERT_pretraining_dataset.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_14_util.h
../utils/ch_14_util.cpp
)

target_link_libraries(14_BERT_pretraining_dataset  ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(14_BERT_pretraining_dataset PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(14_PretrainingBERT)

target_sources(14_PretrainingBERT PRIVATE
PretrainingBERT.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_14_util.h
../utils/ch_14_util.cpp
)

target_link_libraries(14_PretrainingBERT ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(14_PretrainingBERT PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)


# ---------------------------------------------------------------
add_executable(14_BERT)

target_sources(14_BERT PRIVATE
BERT.cpp
../utils/ch_8_9_util.h
../utils/ch_8_9_util.cpp
../utils.h 
../utils.cpp
../utils/ch_14_util.h
../utils/ch_14_util.cpp
)

target_link_libraries(14_BERT ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(14_BERT PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)

