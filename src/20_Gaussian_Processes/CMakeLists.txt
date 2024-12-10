
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
add_executable(20_gp_priors)

target_sources(20_gp_priors PRIVATE 
gp_priors.cpp
../utils.h
../utils.cpp
../utils/ch_20_util.h
../utils/ch_20_util.cpp
)

target_link_libraries(20_gp_priors ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(20_gp_priors PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(20_gp_inference)

target_sources(20_gp_inference PRIVATE 
gp_inference.cpp
../utils.h
../utils.cpp
../utils/ch_20_util.h
../utils/ch_20_util.cpp
)

target_link_libraries(20_gp_inference ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(20_gp_inference PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


