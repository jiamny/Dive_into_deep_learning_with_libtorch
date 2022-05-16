
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

# ---------------------------------------------------------------
add_executable(18_IntegralCalculus)

target_sources(18_IntegralCalculus PRIVATE 
IntegralCalculus.cpp
)

target_include_directories(18_IntegralCalculus PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_IntegralCalculus ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_IntegralCalculus PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable(18_LinearAlgebraicOperations)

target_sources(18_LinearAlgebraicOperations PRIVATE 
LinearAlgebraicOperations.cpp
../utils.h
../utils.cpp
../fashion.h
../fashion.cpp
)

target_include_directories(18_LinearAlgebraicOperations PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_LinearAlgebraicOperations  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_LinearAlgebraicOperations PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable( 18_MaximumLikelihood )

target_sources( 18_MaximumLikelihood PRIVATE MaximumLikelihood.cpp )

target_include_directories(18_MaximumLikelihood PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_MaximumLikelihood ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_MaximumLikelihood PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable( 18_MultivariableCalculus )

target_sources( 18_MultivariableCalculus PRIVATE MultivariableCalculus.cpp )

target_include_directories(18_MultivariableCalculus PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_MultivariableCalculus ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_MultivariableCalculus PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

# ---------------------------------------------------------------
add_executable( 18_RandomVariables )

target_sources( 18_RandomVariables PRIVATE RandomVariables.cpp )

target_include_directories(18_RandomVariables PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_RandomVariables ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_RandomVariables PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_SingleVariableCalculus )

target_sources( 18_SingleVariableCalculus PRIVATE SingleVariableCalculus.cpp )

target_include_directories(18_SingleVariableCalculus PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(18_SingleVariableCalculus ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(18_SingleVariableCalculus PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)




