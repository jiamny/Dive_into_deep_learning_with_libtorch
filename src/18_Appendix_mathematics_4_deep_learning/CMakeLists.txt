

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

# ---------------------------------------------------------------
add_executable(18_Distributions)

target_sources(18_Distributions PRIVATE Distributions.cpp)

target_link_libraries(18_Distributions ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_Distributions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(18_Eigendecompositions)

target_sources(18_Eigendecompositions PRIVATE Eigendecompositions.cpp)

target_link_libraries(18_Eigendecompositions ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_Eigendecompositions PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(18_InformationTheory)

target_sources(18_InformationTheory PRIVATE 
InformationTheory.cpp
../utils.h
../utils.cpp
)

target_link_libraries(18_InformationTheory ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(18_InformationTheory PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(18_IntegralCalculus)

target_sources(18_IntegralCalculus PRIVATE 
IntegralCalculus.cpp
)

target_link_libraries(18_IntegralCalculus ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_IntegralCalculus PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable(18_LinearAlgebraicOperations)

target_sources(18_LinearAlgebraicOperations PRIVATE 
LinearAlgebraicOperations.cpp
../fashion.h
../fashion.cpp
)

target_link_libraries(18_LinearAlgebraicOperations  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(18_LinearAlgebraicOperations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_MaximumLikelihood )

target_sources( 18_MaximumLikelihood PRIVATE MaximumLikelihood.cpp )

target_link_libraries(18_MaximumLikelihood ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_MaximumLikelihood PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_MultivariableCalculus )

target_sources( 18_MultivariableCalculus PRIVATE MultivariableCalculus.cpp )

target_link_libraries(18_MultivariableCalculus ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_MultivariableCalculus PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_NaiveBayes )

target_sources( 18_NaiveBayes PRIVATE 
NaiveBayes.cpp
../fashion.h
../fashion.cpp 
)

target_link_libraries(18_NaiveBayes ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_NaiveBayes PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_RandomVariables )

target_sources( 18_RandomVariables PRIVATE RandomVariables.cpp )

target_link_libraries(18_RandomVariables ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_RandomVariables PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_SingleVariableCalculus )

target_sources( 18_SingleVariableCalculus PRIVATE SingleVariableCalculus.cpp )

target_link_libraries(18_SingleVariableCalculus ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_SingleVariableCalculus PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ---------------------------------------------------------------
add_executable( 18_Statistics )

target_sources( 18_Statistics PRIVATE Statistics.cpp )

target_link_libraries(18_Statistics ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(18_Statistics PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


