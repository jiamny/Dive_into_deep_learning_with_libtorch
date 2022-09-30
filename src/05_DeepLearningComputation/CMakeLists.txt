cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

# ---------------------------------------------------------

add_executable(05_custom_layers)

target_sources(05_custom_layers PRIVATE custom_layers.cpp)

target_link_libraries(05_custom_layers ${TORCH_LIBRARIES})

set_target_properties(05_custom_layers PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------

add_executable(05_model_construction)

target_sources(05_model_construction PRIVATE model_construction.cpp)

target_link_libraries(05_model_construction ${TORCH_LIBRARIES})

set_target_properties(05_model_construction PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------

add_executable(05_ParameterManagement)

target_sources(05_ParameterManagement PRIVATE ParameterManagement.cpp)

target_link_libraries(05_ParameterManagement ${TORCH_LIBRARIES})

set_target_properties(05_ParameterManagement PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------

add_executable(05_File_io)

target_sources(05_File_io PRIVATE File_io.cpp)

target_link_libraries(05_File_io ${TORCH_LIBRARIES})

set_target_properties(05_File_io PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------

add_executable(05_gpu)

target_sources(05_gpu PRIVATE gpu.cpp)

target_link_libraries(05_gpu ${TORCH_LIBRARIES})

set_target_properties(05_gpu PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)


