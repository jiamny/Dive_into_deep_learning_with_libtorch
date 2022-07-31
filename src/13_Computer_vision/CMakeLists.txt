cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.9")
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
add_executable(13_AnchorBoxes)
target_sources(13_AnchorBoxes PRIVATE 
AnchorBoxes.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
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
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_include_directories(13_BoundingBoxes  PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(13_BoundingBoxes  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(13_BoundingBoxes  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
#add_executable(13_TestTorchVision)
#target_sources(13_TestTorchVision PRIVATE TestTorchVision.cpp )
													
#target_include_directories(13_TestTorchVision  PRIVATE 
#${TORCHVISION_INCLUDE_DIRS}
#${Python3_INCLUDE_DIRS}  
#${Python3_NumPy_INCLUDE_DIRS}
#)

#target_link_libraries(13_TestTorchVision  
#${OpenCV_LIBS} 
#${TORCH_LIBRARIES} 
#${TORCHVISION_LIBRARIES}
#${requiredlibs} Python3::Python Python3::NumPy
#)

#set_target_properties(13_TestTorchVision  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_MultiscaleObjectDetection)
target_sources(13_MultiscaleObjectDetection PRIVATE 
MultiscaleObjectDetection.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_include_directories(13_MultiscaleObjectDetection  PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(13_MultiscaleObjectDetection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(13_MultiscaleObjectDetection  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_ObjectDetectionDataset)
target_sources(13_ObjectDetectionDataset PRIVATE 
ObjectDetectionDataset.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_include_directories(13_ObjectDetectionDataset  PRIVATE 
#${TORCHVISION_INCLUDE_DIRS}
${Python3_INCLUDE_DIRS} 
${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(13_ObjectDetectionDataset  
${OpenCV_LIBS} 
${TORCH_LIBRARIES} 
#${TORCHVISION_LIBRARIES}
${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(13_ObjectDetectionDataset  PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(13_FullyConvolutionalNetworks)
target_sources(13_FullyConvolutionalNetworks PRIVATE 
FullyConvolutionalNetworks.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_include_directories(13_ObjectDetectionDataset PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_FullyConvolutionalNetworks ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_FullyConvolutionalNetworks PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_TransposedConvolution)
target_sources(13_TransposedConvolution PRIVATE 
TransposedConvolution.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_include_directories(13_TransposedConvolution PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_TransposedConvolution ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_TransposedConvolution PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_ImageAugmentation)
target_sources(13_ImageAugmentation PRIVATE 
ImageAugmentation.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_include_directories(13_ImageAugmentation PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_ImageAugmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_ImageAugmentation PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_Kaggle_cifar10)
target_sources(13_Kaggle_cifar10 PRIVATE 
Kaggle_cifar10.cpp
../utils.h 
../utils.cpp
../utils/dataloader.hpp
../utils/dataloader.cpp
../utils/datasets.hpp
../utils/datasets.cpp
../utils/transforms.hpp
../utils/transforms.cpp
)

target_include_directories(13_Kaggle_cifar10 PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_Kaggle_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_Kaggle_cifar10 PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_Kaggle_dog)
target_sources(13_Kaggle_dog PRIVATE 
Kaggle_dog.cpp
../utils.h 
../utils.cpp
../utils/dataloader.hpp
../utils/dataloader.cpp
../utils/datasets.hpp
../utils/datasets.cpp
../utils/transforms.hpp
../utils/transforms.cpp
)

target_include_directories(13_Kaggle_dog PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_Kaggle_dog ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_Kaggle_dog PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_SemanticSegmentation)
target_sources(13_SemanticSegmentation PRIVATE 
SemanticSegmentation.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_include_directories(13_SemanticSegmentation PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_SemanticSegmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_SemanticSegmentation PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_NeuralStyleTransfer)
target_sources(13_NeuralStyleTransfer PRIVATE 
NeuralStyleTransfer.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_include_directories(13_NeuralStyleTransfer PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_NeuralStyleTransfer ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_NeuralStyleTransfer PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_Yolo4)
target_sources(13_Yolo4 PRIVATE 
./yolov4/src/main.cc
./yolov4/src/darknet.cc
./yolov4/src/darknet.h
./yolov4/src/config.cc	
./yolov4/src/config.h
./yolov4/src/coco_names.h
#../utils.h
#../utils.cpp
)

target_include_directories(13_Yolo4 PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS} )
target_link_libraries( 13_Yolo4 ${OpenCV_LIBS} ${TORCH_LIBRARIES} 
					   ${requiredlibs} Python3::Python Python3::NumPy)
					   
set_target_properties( 13_Yolo4 PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

