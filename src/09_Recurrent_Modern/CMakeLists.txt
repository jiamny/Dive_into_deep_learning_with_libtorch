
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(09_Gated_recurrent_units)
target_sources(09_Gated_recurrent_units PRIVATE Gated_recurrent_units.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)

target_link_libraries(09_Gated_recurrent_units ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_Gated_recurrent_units PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(09_Long_short_term_memory)
target_sources(09_Long_short_term_memory PRIVATE Long_short_term_memory.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(09_Long_short_term_memory ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_Long_short_term_memory PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(09_Deep_recurrent_neural_networks)
target_sources(09_Deep_recurrent_neural_networks PRIVATE Deep_recurrent_neural_networks.cpp 
					../utils/ch_8_9_util.h
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(09_Deep_recurrent_neural_networks ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_Deep_recurrent_neural_networks PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(09_Bidirectional_RNN)
target_sources(09_Bidirectional_RNN PRIVATE Bidirectional_RNN.cpp 
					../utils/ch_8_9_util.h 
					../utils/ch_8_9_util.cpp
					)
													
target_link_libraries(09_Bidirectional_RNN ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_Bidirectional_RNN PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(09_Machine_translation_dataset)
target_sources(09_Machine_translation_dataset PRIVATE Machine_translation_dataset.cpp 
					../utils/ch_8_9_util.h
					../utils/ch_8_9_util.cpp
					../utils.h
					../utils.cpp
					)
													
target_link_libraries(09_Machine_translation_dataset ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_Machine_translation_dataset PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(09_SequenceToSequence)
target_sources(09_SequenceToSequence PRIVATE SequenceToSequence.cpp 
					../utils/ch_8_9_util.h
					../utils/ch_8_9_util.cpp
					../utils.h
					../utils.cpp
					)
													
target_link_libraries(09_SequenceToSequence ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(09_SequenceToSequence PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


