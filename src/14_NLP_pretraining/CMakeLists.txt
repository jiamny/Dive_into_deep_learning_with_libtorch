
find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
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

target_include_directories(14_BERT_pretraining_dataset PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(14_BERT_pretraining_dataset ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(14_BERT_pretraining_dataset PROPERTIES CXX_STANDARD 17  CXX_STANDARD_REQUIRED YES)





