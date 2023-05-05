
set(requiredlibs "pthread")
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
													
target_link_libraries(13_AnchorBoxes  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(13_AnchorBoxes  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_BoundingBoxes)
target_sources(13_BoundingBoxes PRIVATE 
BoundingBoxes.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_link_libraries(13_BoundingBoxes  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(13_BoundingBoxes  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_MultiscaleObjectDetection)
target_sources(13_MultiscaleObjectDetection PRIVATE 
MultiscaleObjectDetection.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_link_libraries(13_MultiscaleObjectDetection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(13_MultiscaleObjectDetection  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(13_ObjectDetectionDataset)
target_sources(13_ObjectDetectionDataset PRIVATE 
ObjectDetectionDataset.cpp 
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp						
)
													
target_link_libraries(13_ObjectDetectionDataset  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(13_ObjectDetectionDataset  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# -------------------------------------------------------------
add_executable(13_Fine_Tuning)
target_sources(13_Fine_Tuning PRIVATE 
Fine_Tuning.cpp
resnet.h
resnet.cpp
../utils.h 
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp		
../utils/dataloader.hpp
../utils/dataloader.cpp
../utils/datasets.hpp
../utils/datasets.cpp
../utils/transforms.hpp
../utils/transforms.cpp
)

target_link_libraries( 13_Fine_Tuning ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties( 13_Fine_Tuning PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_FullyConvolutionalNetworks)
target_sources(13_FullyConvolutionalNetworks PRIVATE 
FullyConvolutionalNetworks.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_link_libraries( 13_FullyConvolutionalNetworks ${OpenCV_LIBS} ${TORCH_LIBRARIES}  ${requiredlibs} matplot)					   
set_target_properties( 13_FullyConvolutionalNetworks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_TransposedConvolution)
target_sources(13_TransposedConvolution PRIVATE 
TransposedConvolution.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_link_libraries( 13_TransposedConvolution ${OpenCV_LIBS} ${TORCH_LIBRARIES}  ${requiredlibs} )					   
set_target_properties( 13_TransposedConvolution PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_ImageAugmentation)
target_sources(13_ImageAugmentation PRIVATE 
ImageAugmentation.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_link_libraries( 13_ImageAugmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES}  ${requiredlibs} matplot)			   
set_target_properties( 13_ImageAugmentation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

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

target_link_libraries( 13_Kaggle_cifar10 ${OpenCV_LIBS} ${TORCH_LIBRARIES}  ${requiredlibs} matplot)					   
set_target_properties( 13_Kaggle_cifar10 PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

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

target_link_libraries( 13_Kaggle_dog ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)					   
set_target_properties( 13_Kaggle_dog PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_SemanticSegmentation)
target_sources(13_SemanticSegmentation PRIVATE 
SemanticSegmentation.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_link_libraries( 13_SemanticSegmentation ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)					   
set_target_properties( 13_SemanticSegmentation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

# -------------------------------------------------------------
add_executable(13_NeuralStyleTransfer)
target_sources(13_NeuralStyleTransfer PRIVATE 
NeuralStyleTransfer.cpp
../utils.h
../utils.cpp
../utils/ch_13_util.h
../utils/ch_13_util.cpp	
)

target_link_libraries( 13_NeuralStyleTransfer ${OpenCV_LIBS} ${TORCH_LIBRARIES}  ${requiredlibs} matplot)					   
set_target_properties( 13_NeuralStyleTransfer PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES )

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

target_link_libraries( 13_Yolo4 ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )					   
set_target_properties( 13_Yolo4 PROPERTIES CXX_STANDARD 14 CXX_STANDARD_REQUIRED YES )

#-------------------------------------------------------------------------------------
add_executable(13_SingleShotMultiboxDetection )
target_sources(13_SingleShotMultiboxDetection PRIVATE 
		SingleShotMultiboxDetection.cpp
		../utils/ch_13_util.h
		../utils/ch_13_util.cpp	
)

target_link_libraries(13_SingleShotMultiboxDetection  ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(13_SingleShotMultiboxDetection  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

get_filename_component(_fullpath "/media/stree/localssd/LibtorchPrgs/Dive_into_deep_learning_with_libtorch/torchvision" REALPATH)
if(EXISTS "${_fullpath}")

	# -------------------------------------------------------------------------------------------
	# where to find torchvision include and lib files
	# -------------------------------------------------------------------------------------------
	list(APPEND CMAKE_PREFIX_PATH "${_fullpath}")

	set(TORCHVISION_INCLUDE_DIRS "${_fullpath}/include")
	set(TORCHVISION_LIBRARIES "${_fullpath}/lib/libtorchvision.so")

	message(STATUS "    ${CMAKE_PREFIX_PATH}")
	message(STATUS "    libraries: ${TORCHVISION_LIBRARIES}")
	message(STATUS "    include path: ${TORCHVISION_INCLUDE_DIRS}") 


	#----------------------------------------------------------------------------------
	add_executable(13_TestTorchVision)
	target_sources(13_TestTorchVision PRIVATE TestTorchVision.cpp )
													
	target_include_directories(13_TestTorchVision  PRIVATE 
		${TORCHVISION_INCLUDE_DIRS}
	)

	target_link_libraries(13_TestTorchVision  
		${OpenCV_LIBS} 
		${TORCH_LIBRARIES} 
		${TORCHVISION_LIBRARIES}
	)

	set_target_properties(13_TestTorchVision  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

	# -------------------------------------------------------------
	add_executable(13_Region_based_CNNs )
	target_sources(13_Region_based_CNNs PRIVATE RegionBasedCNNs.cpp)

	target_include_directories(13_Region_based_CNNs  PRIVATE 
		${TORCHVISION_INCLUDE_DIRS}
	)

	target_link_libraries(13_Region_based_CNNs  
		${OpenCV_LIBS} 
		${TORCH_LIBRARIES} 
		${TORCHVISION_LIBRARIES}
		${requiredlibs}
	)

	set_target_properties(13_Region_based_CNNs  PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

endif()