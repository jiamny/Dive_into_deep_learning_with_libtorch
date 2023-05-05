
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(08_SequenceModels)

target_sources(08_SequenceModels PRIVATE SequenceModels.cpp 
					../utils.h 
					../utils.cpp
					)
													
target_link_libraries(08_SequenceModels ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(08_SequenceModels PROPERTIES  CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_TextPreprocessing)

target_sources(08_TextPreprocessing PRIVATE TextPreprocessing.cpp 
					..//utils/ch_8_9_util.h 
					..//utils/ch_8_9_util.cpp
					)
													
target_link_libraries(08_TextPreprocessing ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )

set_target_properties(08_TextPreprocessing PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_Language_models)

target_sources(08_Language_models PRIVATE Language_models.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(08_Language_models ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(08_Language_models PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_RecurrentNeuralNetwork)

target_sources(08_RecurrentNeuralNetwork PRIVATE RecurrentNeuralNetwork.cpp 
						../utils/ch_8_9_util.h 
						../utils/ch_8_9_util.cpp
						)

target_link_libraries(08_RecurrentNeuralNetwork ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(08_RecurrentNeuralNetwork PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------

add_executable(08_rnn_concise)

target_sources(08_rnn_concise PRIVATE rnn_concise.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)

target_link_libraries(08_rnn_concise ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(08_rnn_concise PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_rnn_scratch)

target_sources(08_rnn_scratch PRIVATE rnn_scratch.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(08_rnn_scratch ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(08_rnn_scratch PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_rnn_example)

target_sources(08_rnn_example PRIVATE rnn_example.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(08_rnn_example ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(08_rnn_example PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(08_RNN)

target_sources(08_RNN PRIVATE RNN.cpp)
													
target_link_libraries(08_RNN ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(08_RNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
