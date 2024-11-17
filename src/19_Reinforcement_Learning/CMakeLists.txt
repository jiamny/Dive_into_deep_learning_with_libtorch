
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
add_executable(19_Q_learning)

target_sources(19_Q_learning PRIVATE 
Q_learning.cpp
../utils.h
../utils.cpp
)

target_link_libraries(19_Q_learning  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs})
set_target_properties(19_Q_learning PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
