
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
add_executable(16_Matrix_factorization)

target_sources(16_Matrix_factorization PRIVATE 
Matrix_factorization.cpp
../utils.h
../utils.cpp
../utils/ch_16_util.h
../utils/ch_16_util.cpp
)

target_link_libraries(16_Matrix_factorization ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(16_Matrix_factorization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(16_LatentFactorModel)

target_sources(16_LatentFactorModel PRIVATE 
LatentFactorModel.cpp
../utils.h
../utils.cpp
../utils/ch_16_util.h
../utils/ch_16_util.cpp
)

target_link_libraries(16_LatentFactorModel ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(16_LatentFactorModel PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


