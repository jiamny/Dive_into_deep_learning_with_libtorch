

set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(12_AsynchronousComputation)
target_sources(12_AsynchronousComputation PRIVATE 
AsynchronousComputation.cpp 
../utils.h
../utils.cpp						
)
													
target_link_libraries(12_AsynchronousComputation ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(12_AsynchronousComputation PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)


#----------------------------------------------------------------------------------
add_executable(12_Hybridize)
target_sources(12_Hybridize PRIVATE Hybridize.cpp )
													
target_link_libraries(12_Hybridize ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} )
set_target_properties(12_Hybridize PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)



