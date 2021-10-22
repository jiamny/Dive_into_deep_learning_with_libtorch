cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.6m")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")

#----------------------------------------------------------------------------------

add_executable(04_dropout)

target_sources(04_dropout PRIVATE dropout.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_dropout PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_dropout ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_dropout PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_mlp)

target_sources(04_mlp PRIVATE mlp.cpp 
									../utils.h 
									../utils.cpp
									)
													
target_include_directories(04_mlp PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_mlp ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_mlp PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_mlp_consice)

target_sources(04_mlp_consice PRIVATE mlp_consice.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_mlp_consice PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_mlp_consice ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_mlp_consice PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_mlp_scratch)

target_sources(04_mlp_scratch PRIVATE mlp_scratch.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_mlp_scratch PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_mlp_scratch ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_mlp_scratch PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_numerical_stability_initialization)

target_sources(04_numerical_stability_initialization PRIVATE numerical_stability_initialization.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_numerical_stability_initialization PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_numerical_stability_initialization ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_numerical_stability_initialization PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_underfit_overfit)

target_sources(04_underfit_overfit PRIVATE underfit_overfit.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_underfit_overfit PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_underfit_overfit ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_underfit_overfit PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_weight_decay)

target_sources(04_weight_decay PRIVATE weight_decay.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_weight_decay PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_weight_decay ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_weight_decay PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(04_Kaggle_house_prices)

target_sources(04_Kaggle_house_prices PRIVATE Kaggle_house_prices.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(04_Kaggle_house_prices PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(04_Kaggle_house_prices ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(04_weight_decay PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)
