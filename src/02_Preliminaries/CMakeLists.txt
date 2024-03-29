
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")


add_executable(02_autograd)

target_sources(02_autograd PRIVATE autograd.cpp)

target_link_libraries(02_autograd ${TORCH_LIBRARIES})
set_target_properties(02_autograd PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(02_calculus)

target_sources(02_calculus PRIVATE calculus.cpp)

target_link_libraries(02_calculus ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(02_calculus PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(02_data_preprocessing)

target_sources(02_data_preprocessing PRIVATE data_preprocessing.cpp)

target_link_libraries(02_data_preprocessing ${TORCH_LIBRARIES})
set_target_properties(02_data_preprocessing PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(02_ndarray)

target_sources(02_ndarray PRIVATE ndarray.cpp)

target_link_libraries(02_ndarray ${TORCH_LIBRARIES})
set_target_properties(02_ndarray PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(02_linear_algebra)

target_sources(02_linear_algebra PRIVATE linear_algebra.cpp)

target_link_libraries(02_linear_algebra ${TORCH_LIBRARIES})
set_target_properties(02_linear_algebra PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(02_probability)

target_sources(02_probability PRIVATE probability.cpp)

target_link_libraries(02_probability ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(02_probability PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
