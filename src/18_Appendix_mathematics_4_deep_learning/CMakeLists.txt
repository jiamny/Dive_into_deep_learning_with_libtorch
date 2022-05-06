
find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.9")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

# ---------------------------------------------------------------
add_executable(18_Distributions)

target_sources(18_Distributions PRIVATE Distributions.cpp)

target_include_directories(18_Distributions PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_Distributions ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_Distributions PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable(18_Eigendecompositions)

target_sources(18_Eigendecompositions PRIVATE Eigendecompositions.cpp)

target_include_directories(18_Eigendecompositions PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_Eigendecompositions ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_Eigendecompositions PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable(18_InformationTheory)

target_sources(18_InformationTheory PRIVATE 
InformationTheory.cpp
../utils.h
../utils.cpp
)

target_include_directories(18_InformationTheory PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_InformationTheory ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_InformationTheory PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)




