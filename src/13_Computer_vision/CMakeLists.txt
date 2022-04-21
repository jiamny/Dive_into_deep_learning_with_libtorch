cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.9")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(13_AnchorBoxes)
target_sources(13_AnchorBoxes PRIVATE 
AnchorBoxes.cpp 
../utils.h
../utils.cpp
../utils/Ch_13_util.h
../utils/Ch_13_util.cpp						
)
													
target_include_directories(13_AnchorBoxes  PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(13_AnchorBoxes  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(13_AnchorBoxes  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_BoundingBoxes)
target_sources(13_BoundingBoxes PRIVATE 
BoundingBoxes.cpp 
../utils.h
../utils.cpp
../utils/Ch_13_util.h
../utils/Ch_13_util.cpp						
)
													
target_include_directories(13_BoundingBoxes  PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(13_BoundingBoxes  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(13_BoundingBoxes  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_TestTorchVision)
target_sources(13_TestTorchVision PRIVATE TestTorchVision.cpp )
													
target_include_directories(13_TestTorchVision  PRIVATE 
${TORCHVISION_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS}  
${Python3_NumPy_INCLUDE_DIRS}
)

target_link_libraries(13_TestTorchVision  
${OpenCV_LIBS} ${TORCH_LIBRARIES} ${TORCHVISION_LIBRARIES}
${requiredlibs} Python3::Python Python3::NumPy
)

set_target_properties(13_TestTorchVision  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)


