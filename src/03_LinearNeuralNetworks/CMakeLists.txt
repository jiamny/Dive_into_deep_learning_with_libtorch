set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------

add_executable(03_linear_regression)

target_sources(03_linear_regression PRIVATE linear_regression.cpp 
													../utils.h 
													../utils.cpp
													)
													
target_link_libraries(03_linear_regression ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_linear_regression PROPERTIES  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(03_linear_regression_concise)

target_sources(03_linear_regression_concise PRIVATE linear_regression_concise.cpp 
													../utils.h 
													../utils.cpp
													)
													
target_link_libraries(03_linear_regression_concise ${TORCH_LIBRARIES} ${requiredlibs})											
set_target_properties(03_linear_regression_concise PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(03_linear_regression_scratch)

target_sources(03_linear_regression_scratch PRIVATE linear_regression_scratch.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_link_libraries(03_linear_regression_scratch ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_linear_regression_scratch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(03_load_image_dataset)

target_sources(03_load_image_dataset PRIVATE load_image_dataset.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_link_libraries(03_load_image_dataset ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_load_image_dataset PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(03_softmax_regression_concise)

target_sources(03_softmax_regression_concise PRIVATE softmax_regression_concise.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_link_libraries(03_softmax_regression_concise ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_softmax_regression_concise PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(03_softmax_regression_scratch)

target_sources(03_softmax_regression_scratch PRIVATE softmax_regression_scratch.cpp 
													../utils.h 
													../utils.cpp
													../fashion.cpp
													../fashion.h)
													
target_link_libraries(03_softmax_regression_scratch ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(03_softmax_regression_scratch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



