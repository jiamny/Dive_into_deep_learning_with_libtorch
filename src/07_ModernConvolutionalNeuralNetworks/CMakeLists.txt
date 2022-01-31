cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.8")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "png")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    list(APPEND requiredlibs "OpenMP::OpenMP_CXX")
endif()

#----------------------------------------------------------------------------------

add_executable(07_AlexNet)

target_sources(07_AlexNet PRIVATE AlexNet.cpp 
									../utils.h 
									../utils.cpp
									../utils/dataloader.hpp
									../utils/dataloader.cpp
									../utils/datasets.hpp
									../utils/datasets.cpp
									../utils/transforms.hpp
									../utils/transforms.cpp
									)
													
target_include_directories(07_AlexNet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_AlexNet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_AlexNet PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_batch_norm)

target_sources(07_batch_norm PRIVATE batch_norm.cpp 
									../utils.h 
									../utils.cpp
									../fashion.h
									../fashion.cpp
									)
													
target_include_directories(07_batch_norm PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_batch_norm ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_batch_norm PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_DensNet)

target_sources(07_DensNet PRIVATE DensNet.cpp 
									../utils.h 
									../utils.cpp
								    ../utils/dataloader.hpp
									../utils/dataloader.cpp
									../utils/datasets.hpp
									../utils/datasets.cpp
									../utils/transforms.hpp
									../utils/transforms.cpp
									)
													
target_include_directories(07_DensNet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_DensNet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_DensNet PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_GoogLeNet)

target_sources(07_GoogLeNet PRIVATE GoogLeNet.cpp 
									../utils.h 
									../utils.cpp
								    ../utils/dataloader.hpp
									../utils/dataloader.cpp
									../utils/datasets.hpp
									../utils/datasets.cpp
									../utils/transforms.hpp
									../utils/transforms.cpp
									)
													
target_include_directories(07_GoogLeNet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_GoogLeNet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_GoogLeNet PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_NiN)

target_sources(07_NiN PRIVATE NiN.cpp 
									../utils.h 
									../utils.cpp
								    ../utils/dataloader.hpp
									../utils/dataloader.cpp
									../utils/datasets.hpp
									../utils/datasets.cpp
									../utils/transforms.hpp
									../utils/transforms.cpp
									)
													
target_include_directories(07_NiN PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_NiN ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_NiN PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_ResNet)

target_sources(07_ResNet PRIVATE ResNet.cpp 
										../utils.h 
										../utils.cpp
								    	../utils/dataloader.hpp
										../utils/dataloader.cpp
										../utils/datasets.hpp
										../utils/datasets.cpp
										../utils/transforms.hpp
										../utils/transforms.cpp
										)
														
target_include_directories(07_ResNet PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_ResNet ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_ResNet PROPERTIES
	  CXX_STANDARD 14
	  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(07_VGG)

target_sources(07_VGG PRIVATE VGG.cpp 
										../utils.h 
										../utils.cpp
								    	../utils/dataloader.hpp
										../utils/dataloader.cpp
										../utils/datasets.hpp
										../utils/datasets.cpp
										../utils/transforms.hpp
										../utils/transforms.cpp
										)
														
target_include_directories(07_VGG PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(07_VGG ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(07_VGG PROPERTIES
	  CXX_STANDARD 14
	  CXX_STANDARD_REQUIRED YES
)


