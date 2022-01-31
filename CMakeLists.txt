cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

project(D2L_pytorch_cpp VERSION 1.0.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(PYTORCH_VERSION "1.10.1")

set(CMAKE_CXX_STANDARD 14)

#--------------------------------------------------------------------------------------------
# following line solve the issue of Can't link a project using cmake with OpenCV and LibTorch
# -------------------------------------------------------------------------------------------
list(APPEND CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake")               

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/libtorch/share/cmake/Torch")                  

# add torch package
find_package(Torch ${PYTORCH_VERSION} EXACT QUIET PATHS "${CMAKE_SOURCE_DIR}/libtorch") 

message(STATUS "Torch library status:")
message(STATUS "    version: ${Torch_VERSION}")
message(STATUS "    libraries: ${TORCH_LIBRARIES}")
message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}") 

# add opencv package to the project
find_package( OpenCV 4.5.3 REQUIRED PATHS "/usr/local/")
#find_package( OpenCV REQUIRED )
INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS} ) 
SET(OpenCV_LIBRARIES ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -lm -ldl")

if(NOT Torch_FOUND)
    unset(Torch_FOUND)
    include(fetch_libtorch)
endif()

INCLUDE_DIRECTORIES( ${TORCH_INCLUDE_DIRS} )

#find_package(Python 3 COMPONENTS Interpreter Development NumPy REQUIRED)

#add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

#message("Python_FOUND:${Python_FOUND}")
#message("Python_VERSION:${Python_VERSION}")
#message("Python_Development_FOUND:${Python_Development_FOUND}")
#message("Python_LIBRARIES:${Python_LIBRARIES}")
#message("{Python_INCLUDE_DIRS:${Python_INCLUDE_DIRS}")
#message("{Python_NumPy_INCLUDE_DIRS:${Python_NumPy_INCLUDE_DIRS}")
#message("{Python_NumPy_LIBRARIES: ${PYTHON_LIBRARIES} Python::Python Python::NumPy")

# Add  sub-projects:
add_subdirectory("src/02_Preliminaries")	
add_subdirectory("src/03_LinearNeuralNetworks")
add_subdirectory("src/04_MultilayerPerceptrons")
add_subdirectory("src/05_DeepLearningComputation")
add_subdirectory("src/06_ConvolutionalNeuralNetworks")
add_subdirectory("src/07_ModernConvolutionalNeuralNetworks")
add_subdirectory("src/08_RecurrentNeuralNetworks")
add_subdirectory("src/09_Recurrent_Modern")
add_subdirectory("src/utils")

if(MSVC)
    include(copy_torch_dlls)
    copy_torch_dlls(${EXECUTABLE_NAME})
endif()
