
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(04_dropout)

target_sources(04_dropout PRIVATE dropout.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_dropout ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_dropout PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_mlp)

target_sources(04_mlp PRIVATE mlp.cpp 
									../utils.h 
									../utils.cpp
									)
													
target_link_libraries(04_mlp ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_mlp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_mlp_consice)

target_sources(04_mlp_consice PRIVATE mlp_consice.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_mlp_consice ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_mlp_consice PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_mlp_scratch)

target_sources(04_mlp_scratch PRIVATE mlp_scratch.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_mlp_scratch ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_mlp_scratch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_numerical_stability_initialization)

target_sources(04_numerical_stability_initialization PRIVATE numerical_stability_initialization.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_numerical_stability_initialization ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_numerical_stability_initialization PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_underfit_overfit)

target_sources(04_underfit_overfit PRIVATE underfit_overfit.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_underfit_overfit ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_underfit_overfit PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_weight_decay)

target_sources(04_weight_decay PRIVATE weight_decay.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_link_libraries(04_weight_decay ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_weight_decay PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(04_Kaggle_house_prices)

target_sources(04_Kaggle_house_prices PRIVATE Kaggle_house_prices.cpp 
									../utils.h 
									../utils.cpp
									)
													
target_link_libraries(04_Kaggle_house_prices ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(04_Kaggle_house_prices PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
