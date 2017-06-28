PROJECT(Subdiv-Fitting-Tests)

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
if(COMMAND cmake_policy)
      cmake_policy(SET CMP0003 NEW)  # Required by CMake 2.7+
endif(COMMAND cmake_policy)

LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

# --------------------------------------------------------------------------
#   The list of "libs" which can be included can be found in:
#     http://www.mrpt.org/Libraries
#
#   The dependencies of a library are automatically added, so you only 
#    need to specify the top-most libraries your code depend on.
# --------------------------------------------------------------------------
FIND_PACKAGE(MRPT REQUIRED base gui opengl maps obs)
FIND_PACKAGE(OpenCV REQUIRED core highgui imgproc imgcodecs)
FIND_PACKAGE(OpenNI2 REQUIRED)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${OpenNI2_INCLUDE_DIRS})

#Include/link OpenSubDiv
#Warning!! If MSVC, include <iso646.h> within the code to define words like "and", "not"... as operators.
#SET(osd_PATH_DEFAULT "C:/Users/Mariano/Programas/OpenSubdiv nuevo")
SET(osd_PATH_DEFAULT "C:/Users/jaimez/programs/openSubdiv")
SET(osd_PATH "${osd_PATH_DEFAULT}" CACHE PATH "Path to osd")
SET(osd_INCLUDE_DIRS "${osd_PATH}/source")
SET(osd_LIBRARY_DIRS "${osd_PATH}/build/lib/Release")
SET(osd_LIB "${osd_LIBRARY_DIRS}/osdCPU.lib")

INCLUDE_DIRECTORIES(${osd_INCLUDE_DIRS})
LINK_DIRECTORIES(${osd_LIBRARY_DIRS})

INCLUDE_DIRECTORIES($ENV{OPENNI2_INCLUDE})
LINK_DIRECTORIES($ENV{OPENNI2_LIB})


#Set the projects
#---------------------------------------------------------------			

ADD_EXECUTABLE(3D-Model-Fitting
	main_3D_model_fitting.cpp
	3D_model_fitting.h
	3D_model_fitting.cpp
	3D_model_tracking.cpp
	3D_model_fitting_solvers.cpp
	3D_model_fitting_visualization.cpp
	3D_model_fitting_segmentation.cpp
	3D_model_fitting_io.cpp)
	
TARGET_LINK_LIBRARIES(3D-Model-Fitting
		${MRPT_LIBS}
		${OpenCV_LIBS}
		${osd_LIB})
		

ADD_EXECUTABLE(3D-Model-Tracking
	main_3D_model_tracking.cpp
	3D_model_fitting.h
	3D_model_fitting.cpp
	3D_model_tracking.cpp
	3D_model_fitting_solvers.cpp
	3D_model_fitting_visualization.cpp
	3D_model_fitting_segmentation.cpp
	3D_model_fitting_io.cpp)
	
TARGET_LINK_LIBRARIES(3D-Model-Tracking
		${MRPT_LIBS}
		${OpenCV_LIBS}
		${osd_LIB})
		
		
ADD_EXECUTABLE(KinectFusion-Datatest
	main_KinectFusion_datatest.cpp
	KinectFusion_datatest.cpp
	KinectFusion_datatest.h)
	
TARGET_LINK_LIBRARIES(KinectFusion-Datatest
		${MRPT_LIBS}
		${OpenCV_LIBS}
		${osd_LIB})


ADD_EXECUTABLE(Image-Capturer
	main_image_capturer.cpp)
	
TARGET_LINK_LIBRARIES(Image-Capturer
		${MRPT_LIBS}
		${OpenCV_LIBS}
		${OpenNI2_LIBRARY})
		
		
ADD_EXECUTABLE(Test-Eigen-Sparse
	main_test_eigen_sparse.cpp)
	
TARGET_LINK_LIBRARIES(Test-Eigen-Sparse
		${MRPT_LIBS})		


IF(CMAKE_COMPILER_IS_GNUCXX)
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mtune=native")
ENDIF(CMAKE_COMPILER_IS_GNUCXX)




