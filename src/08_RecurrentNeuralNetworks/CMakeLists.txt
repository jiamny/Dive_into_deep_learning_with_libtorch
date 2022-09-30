cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

find_package(Python3 COMPONENTS Development NumPy)

add_definitions(-DMATPLOTLIBCPP_PYTHON_HEADER=Python.h)

set(requiredlibs "python3.10")
list(APPEND requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------

add_executable(08_SequenceModels)

target_sources(08_SequenceModels PRIVATE SequenceModels.cpp 
					../utils.h 
					../utils.cpp
					)
													
target_include_directories(08_SequenceModels PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_SequenceModels ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_SequenceModels PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_TextPreprocessing)

target_sources(08_TextPreprocessing PRIVATE TextPreprocessing.cpp 
					..//utils/ch_8_9_util.h 
					..//utils/ch_8_9_util.cpp
					)
													
target_include_directories(08_TextPreprocessing PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_TextPreprocessing ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_TextPreprocessing PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_Language_models)

target_sources(08_Language_models PRIVATE Language_models.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_include_directories(08_Language_models PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_Language_models ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_Language_models PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_RecurrentNeuralNetwork)

target_sources(08_RecurrentNeuralNetwork PRIVATE RecurrentNeuralNetwork.cpp 
						../utils/ch_8_9_util.h 
						../utils/ch_8_9_util.cpp
						)
													
target_include_directories(08_RecurrentNeuralNetwork PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_RecurrentNeuralNetwork ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_RecurrentNeuralNetwork PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_rnn_concise)

target_sources(08_rnn_concise PRIVATE rnn_concise.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_include_directories(08_rnn_concise PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_rnn_concise ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_rnn_concise PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_rnn_scratch)

target_sources(08_rnn_scratch PRIVATE rnn_scratch.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_include_directories(08_rnn_scratch PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_rnn_scratch ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_rnn_scratch PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_rnn_example)

target_sources(08_rnn_example PRIVATE rnn_example.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_include_directories(08_rnn_example PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_rnn_example ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_rnn_example PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

#----------------------------------------------------------------------------------

add_executable(08_RNN)

target_sources(08_RNN PRIVATE RNN.cpp)
													
target_include_directories(08_RNN PRIVATE ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
target_link_libraries(08_RNN ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} Python3::Python Python3::NumPy)

set_target_properties(08_RNN PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED YES
)

