

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
add_executable(Utils_img_loder_demo)

target_sources(Utils_img_loder_demo PRIVATE img_loder_demo.cpp 
						../utils.h 
						../utils.cpp
						ch_13_util.h
						ch_13_util.cpp
						dataloader.hpp
						dataloader.cpp
						datasets.hpp
						datasets.cpp
						transforms.hpp
						transforms.cpp
						)
													
target_link_libraries(Utils_img_loder_demo ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(Utils_img_loder_demo PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------------
add_executable(Utils_matplot_plot)

target_sources(Utils_matplot_plot PRIVATE Matplot_plot.cpp)
													
target_link_libraries(Utils_matplot_plot ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${requiredlibs} matplot)
set_target_properties(Utils_matplot_plot PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# ------------------------------------------------------------------------
add_executable(Utils_matplot_exp)

target_sources(Utils_matplot_exp PRIVATE matplot_exp.cpp)
													
target_link_libraries(Utils_matplot_exp ${requiredlibs} matplot)
set_target_properties(Utils_matplot_exp PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
