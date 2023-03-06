
set(requiredlibs "pthread")
list(APPEND requiredlibs "dl")
list(APPEND requiredlibs "m")
list(APPEND requiredlibs "jsoncpp") # link to libjsoncpp.so/libjsoncpp.a

#----------------------------------------------------------------------------------
add_executable(11_Optimization_intro)

target_sources(11_Optimization_intro PRIVATE Optimization_intro.cpp )

target_link_libraries(11_Optimization_intro ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Optimization_intro PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_MinibatchStochasticGradientDescent)
target_sources(11_MinibatchStochasticGradientDescent PRIVATE 
MinibatchStochasticGradientDescent.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_MinibatchStochasticGradientDescent ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_MinibatchStochasticGradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_Adagrad)
target_sources(11_Adagrad PRIVATE  Adagrad.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_Adagrad ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Adagrad PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_Adadelta)
target_sources(11_Adadelta PRIVATE  Adadelta.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_Adadelta ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Adadelta PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_Adam)
target_sources(11_Adam PRIVATE  Adam.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_Adam ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Adam PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_Convexity)
target_sources(11_Convexity PRIVATE  Convexity.cpp
)

target_link_libraries(11_Convexity ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Convexity PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_GradientDescent)
target_sources(11_GradientDescent PRIVATE  GradientDescent.cpp
)

target_link_libraries(11_GradientDescent ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_GradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_Momentum)
target_sources(11_Momentum PRIVATE  Momentum.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_Momentum ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_Momentum PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_StochasticGradientDescent)
target_sources(11_StochasticGradientDescent PRIVATE  StochasticGradientDescent.cpp
../utils/ch_11_util.h
../utils/ch_11_util.cpp
)

target_link_libraries(11_StochasticGradientDescent ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_StochasticGradientDescent PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_RMSProp)
target_sources(11_RMSProp PRIVATE  RMSProp.cpp
../csvloader.h
../utils/ch_11_util.h
../utils/ch_11_util.cpp
../utils.h
../utils.cpp
)

target_link_libraries(11_RMSProp ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_RMSProp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

#----------------------------------------------------------------------------------
add_executable(11_LearningRateScheduling)
target_sources(11_LearningRateScheduling PRIVATE  LearningRateScheduling.cpp
../utils.h 
../utils.cpp
../fashion.h
../fashion.cpp
)

target_link_libraries(11_LearningRateScheduling ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(11_LearningRateScheduling PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)




