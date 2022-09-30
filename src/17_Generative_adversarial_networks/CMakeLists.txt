
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
add_executable(17_DeepConvolutional_GAN)

target_sources(17_DeepConvolutional_GAN PRIVATE 
DeepConvolutional_GAN.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp
../utils/dataloader.hpp
../utils/dataloader.cpp
../utils/datasets.hpp
../utils/datasets.cpp
../utils/transforms.hpp
../utils/transforms.cpp
)

target_include_directories(17_DeepConvolutional_GAN PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(17_DeepConvolutional_GAN ${OpenCV_LIBS}  ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(17_DeepConvolutional_GAN PROPERTIES CXX_STANDARD 14  CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(17_GenerativeAdversarialNetwork)

target_sources(17_GenerativeAdversarialNetwork PRIVATE 
GenerativeAdversarialNetwork.cpp
../utils.h
../utils.cpp
)

target_include_directories(17_GenerativeAdversarialNetwork PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(17_GenerativeAdversarialNetwork ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(17_GenerativeAdversarialNetwork PROPERTIES CXX_STANDARD 14  CXX_STANDARD_REQUIRED YES)
