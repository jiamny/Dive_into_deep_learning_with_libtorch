
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
add_executable(21_Hyperopt_intro)

target_sources(21_Hyperopt_intro PRIVATE 
Hyperopt_intro.cpp
../utils.h
../utils.cpp
../fashion.cpp
../fashion.h)

target_link_libraries(21_Hyperopt_intro ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(21_Hyperopt_intro PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(21_Hyperopt_parameters_search)

target_sources(21_Hyperopt_parameters_search PRIVATE 
Hyper_parameters_search.cpp
../utils.h
../utils.cpp
../fashion.cpp
../fashion.h)

target_link_libraries(21_Hyperopt_parameters_search ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(21_Hyperopt_parameters_search PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



