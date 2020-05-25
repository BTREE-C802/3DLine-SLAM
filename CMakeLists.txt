#1指定cmake版本
cmake_minimum_required(VERSION 2.8)

#2Check c++11 or 添加c++11标准支持
SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
set (CMAKE_BUILD_TYPE Release)
SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include_directories(${CMAKE_CURRENT_BINARY_DIR})

#3指定项目的名称，一般和项目的文件夹名称对应
project(3DLines-SLAM)
  MESSAGE(STATUS "Project: 3DLines-SLAM")    

#4找到后面需要库和头文件的包
## find OpenCV-------------------------------------------------------------
FIND_PACKAGE(OpenCV REQUIRED)

# check for OpenCV >= 3
if(${OpenCV_VERSION_MAJOR} GREATER "2")
  FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui imgcodecs)
  SET(L3DPP_OPENCV3 1)
else(${OpenCV_VERSION_MAJOR} GREATER "2")
  FIND_PACKAGE(OpenCV REQUIRED COMPONENTS core imgproc highgui)
endif(${OpenCV_VERSION_MAJOR} GREATER "2")

set (EXTRA_INC_DIRS
    ${EXTRA_INC_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
set (EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${OpenCV_LIBS}
)
link_directories(${OPENCV_LIB_DIR})
## find Eigen3 and pangolin -------------------------------------------------------------
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(Eigen3) 
find_package(Pangolin REQUIRED)
include_directories(
${EIGEN3_INCLUDE_DIR}
${Pangolin_INCLUDE_DIRS}
)
set (EXTRA_LIBRARIES
      ${EXTRA_LIBRARIES}
      ${EIGEN3_LIBS}
      ${Pangolin_LIBRARIES}
)
## find BoW2 and g2o -------------------------------------------------------------
set (EXTRA_LIBRARIES
      ${EXTRA_LIBRARIES}
      ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
      ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)
## find pcl -------------------------------------------------------------
find_package( PCL REQUIRED COMPONENT common io )
include_directories( ${PCL_INCLUDE_DIRS} )
add_definitions( ${PCL_DEFINITIONS} )
set (EXTRA_LIBRARIES
    ${EXTRA_LIBRARIES}
    ${PCL_LIBRARIES}
)
## find CUDA -------------------------------------------------------------
  FIND_PACKAGE(CUDA)
  IF(CUDA_FOUND)
    set (EXTRA_INC_DIRS
        ${EXTRA_INC_DIRS}
        ${CUDA_INCLUDE_DIRS}
		${CUDA_SDK_INCLUDE_DIR}
    )
	cuda_include_directories(${EXTRA_INC_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
    set (EXTRA_LIBRARIES
        ${EXTRA_LIBRARIES}
        ${CUDA_LIBS}
        ${CUDA_cublas_LIBRARY}
    )
  ENDIF(CUDA_FOUND)
## find Boost -------------------------------------------------------------
set(BOOST_LIBRARY_DIR "" CACHE PATH " directory of the boost lib")
set(BOOST_INCLUDE_DIR "" CACHE PATH " directory of the boost headers")
SET(BOOST_LIBRARYDIR "${BOOST_LIBRARY_DIR}")
SET(BOOST_INCLUDEDIR "${BOOST_INCLUDE_DIR}")

if ("${BOOST_LIBRARY_DIR}" STREQUAL "" OR "${BOOST_INCLUDE_DIR}" STREQUAL "")
  message("BOOST library dir not set... trying to find it...")  
endif ("${BOOST_LIBRARY_DIR}" STREQUAL "" OR "${BOOST_INCLUDE_DIR}" STREQUAL "")

IF( WIN32 )
  SET(Boost_USE_STATIC_LIBS ON)
ENDIF( WIN32 )

FIND_PACKAGE(Boost REQUIRED COMPONENTS serialization filesystem system thread)

if(Boost_FOUND)
        message("boost include dirs: " ${Boost_INCLUDE_DIRS})
        message("boost libraries:    " ${Boost_LIBRARIES})
        set (EXTRA_INC_DIRS
                ${EXTRA_INC_DIRS}
                ${Boost_INCLUDE_DIRS}
        )

          set (EXTRA_LIBRARIES
                ${EXTRA_LIBRARIES}
                ${Boost_LIBRARIES}
          )
endif(Boost_FOUND)
## find CERES -------------------------------------------------------------
find_package(Ceres)
if(Ceres_FOUND)
  include_directories(${CERES_INCLUDES})
  set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} optimized ceres)
endif(Ceres_FOUND)

if(NOT Ceres_FOUND AND WIN32)

  message("ceres not found, try to define it manually...")

  # set ceres lib and include dir manually
  set(CERES_LIBRARY "" CACHE FILEPATH "path to the ceres.lib")
  set(CERES_ROOT "" CACHE PATH "path to the ceres include dir")
  
  if (NOT "${CERES_LIBRARY}" STREQUAL "" AND NOT "${CERES_ROOT}" STREQUAL "")
     set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} ${CERES_LIBRARY})
     include_directories(${CERES_ROOT})
     message("ceres include dir: " ${CERES_ROOT})
     message("ceres_lib:         " ${CERES_LIBRARY})
  endif (NOT "${CERES_LIBRARY}" STREQUAL "" AND NOT "${CERES_ROOT}" STREQUAL "")

  # only 64bit CERES is included
  if( "${CERES_LIBRARY}" STREQUAL "" AND CMAKE_SIZEOF_VOID_P EQUAL 8 )
  
      message("using precompiled ceres.... (WIN64 only)")
      
      # set glog lib and include dir
      set(GLOG_LIBRARY "" CACHE FILEPATH "path to the glog.lib")
      set(GLOG_ROOT "" CACHE PATH "path to the glog build dir")
    
      if (NOT "${GLOG_LIBRARY}" STREQUAL "" AND NOT "${GLOG_ROOT}" STREQUAL "")
         set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} ${GLOG_LIBRARY})
         include_directories(${GLOG_ROOT})
         message("glog build dir: " ${GLOG_ROOT})
         message("glog_lib:       " ${GLOG_LIBRARY})
      endif (NOT "${GLOG_LIBRARY}" STREQUAL "" AND NOT "${GLOG_ROOT}" STREQUAL "")
    
      # set gflags lib and include dir
      set(GFLAGS_LIBRARY "" CACHE FILEPATH "path to the gflags_*.lib")
      set(GFLAGS_ROOT "" CACHE PATH "path to the gflags include dir")
      
      if (NOT "${GFLAGS_LIBRARY}" STREQUAL "" AND NOT "${GFLAGS_ROOT}" STREQUAL "")
         set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} ${GFLAGS_LIBRARY})
         include_directories(${GFLAGS_ROOT})
         message("gflags build dir: " ${GFLAGS_ROOT})
         message("gflags_lib:       " ${GFLAGS_LIBRARY})
      endif (NOT "${GFLAGS_LIBRARY}" STREQUAL "" AND NOT "${GFLAGS_ROOT}" STREQUAL "")
      
    # use precompiled CERES version
    set(PRECOMPILED_DIR ${CMAKE_SOURCE_DIR}/libs/precompiled)
    
    set(PRECOMPILED_LIB lib64)
    set(PRECOMPILED_LIB_DIR_FILTER bin64/ )
    #deep_copy_file(${PRECOMPILED_DIR} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE} dll ${PRECOMPILED_LIB_DIR_FILTER} release)

    UNSET(CERES_LIBDIR CACHE)
    FIND_PATH(CERES_LIBDIR NAMES ceres.lib PATHS ${PRECOMPILED_DIR}/ceres/${PRECOMPILED_LIB}/vc10)
    FIND_PATH(CERES_INCLUDE_DIR NAMES ceres/ceres.h PATHS ${PRECOMPILED_DIR}/ceres/include)

    if(CERES_LIBDIR AND CERES_INCLUDE_DIR)
      find_package_handle_standard_args(CERES DEFAULT_MSG CERES_INCLUDE_DIR)
      set(EXTRA_LIBRARIES ${EXTRA_LIBRARIES} optimized ceres)
      include_directories(${CERES_INCLUDE_DIR})
      link_directories(${CERES_LIBDIR})
      set(Ceres_FOUND 1)
    endif(CERES_LIBDIR AND CERES_INCLUDE_DIR)
  elseif( "${CERES_LIBRARY}" STREQUAL "" AND CMAKE_SIZEOF_VOID_P EQUAL 8 )
    message("Sorry... precompiled ceres only functioning with 64 bit!")  
  endif( "${CERES_LIBRARY}" STREQUAL "" AND CMAKE_SIZEOF_VOID_P EQUAL 8 )
		   
endif(NOT Ceres_FOUND AND WIN32)
## OpenMP -------------------------------------------------------------
FIND_PACKAGE(OpenMP)
IF(OPENMP_FOUND)
    message("OPENMP_FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
ENDIF(OPENMP_FOUND)
## tclap -------------------------------------------------------------
set(TCLAP_INCLUDE_DIR "" CACHE PATH "include directory of tclap")
if (NOT "${TCLAP_INCLUDE_DIR}" STREQUAL "")
  set (EXTRA_INC_DIRS
                ${EXTRA_INC_DIRS}
                ${TCLAP_INCLUDE_DIR}
        )
endif (NOT "${TCLAP_INCLUDE_DIR}" STREQUAL "")

##-----------------------------------------------------------------------------
## RapidJSON
#FIND_PACKAGE(RapidJSON)

add_definitions(-frounding-math)
add_definitions("-D_FORCE_INLINES") # needed for ubuntu 16.xx

## specify all extra inlcude dirs
include_directories(${EXTRA_INC_DIRS})
## switches for external (optional) libs
IF(CUDA_FOUND)
    option(APP_LINE_3D++_USE_CUDA "Line3D++: use CUDA" ON)
ELSE(CUDA_FOUND)
    option(APP_LINE_3D++_USE_CUDA "Line3D++: use CUDA" OFF)
ENDIF(CUDA_FOUND)

IF(OPENMP_FOUND)
    option(APP_LINE_3D++_USE_OPENMP "Line3D++: use OpenMP" ON)
ELSE(OPENMP_FOUND)
    option(APP_LINE_3D++_USE_OPENMP "Line3D++: use OpenMP" OFF)
ENDIF(OPENMP_FOUND)

IF(Ceres_FOUND)
    option(APP_LINE_3D++_USE_CERES "Line3D++: use Ceres" ON)
ELSE(Ceres_FOUND)
    option(APP_LINE_3D++_USE_CERES "Line3D++: use Ceres" OFF)
ENDIF(Ceres_FOUND)

if(CUDA_FOUND AND APP_LINE_3D++_USE_CUDA)
    SET(L3DPP_CUDA 1)
endif(CUDA_FOUND AND APP_LINE_3D++_USE_CUDA)

if(OPENMP_FOUND AND APP_LINE_3D++_USE_OPENMP)
    SET(L3DPP_OPENMP 1)
endif(OPENMP_FOUND AND APP_LINE_3D++_USE_OPENMP)

if(Ceres_FOUND AND APP_LINE_3D++_USE_CERES)
    SET(L3DPP_CERES 1)
endif(Ceres_FOUND AND APP_LINE_3D++_USE_CERES)

## create header with defines
CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/configLIBS.h.in ${CMAKE_CURRENT_BINARY_DIR}/configLIBS.h)

#---- combine external libs -----
IF(L3DPP_OPENCV3)
  set(ALL_LIBRARIES ${EXTRA_LIBRARIES})
ELSE(L3DPP_OPENCV3)
  set(ALL_LIBRARIES ${EXTRA_LIBRARIES})
ENDIF(L3DPP_OPENCV3)

# 6 设置路径并创建共享库（把工程内的cpp文件都创建成共享库文件，方便通过头文件来调用）
#---- Add Line3D++ library----
SET(Line3D_HEADERS include/Converter.h include/FrameDrawer.h include/Frame.h include/Initializer.h include/KeyFrame.h
include/KeyFrameDatabase.h include/L3DPPing.h include/LocalMapping.h include/LoopClosing.h include/Map.h include/MapPoint.h include/MapDrawer.h include/Optimizer.h
include/ORBextractor.h include/ORBmatcher.h include/PnPsolver.h include/Sim3Solver.h include/System.h include/Tracking.h include/Viewer.h
include/line3D.h include/view.h include/clustering.h include/universe.h include/serialization.h include/commons.h include/dataArray.h include/segment3D.h include/optimization.h 
include/sparsematrix.h include/cudawrapper.h configLIBS.h include/LineDescriptor.hh include/EDLineDetector.hh include/LineStructure.hh)
IF(L3DPP_CUDA)
        SET(Line3D_SOURCES  src/Converter.cc src/FrameDrawer.cc src/Frame.cc  src/Initializer.cc src/KeyFrame.cc
        src/KeyFrameDatabase.cc src/L3DPPing.cpp src/LocalMapping.cc src/LoopClosing.cc src/Map.cc src/MapPoint.cc src/MapDrawer.cc src/Optimizer.cc
        src/ORBextractor.cc src/ORBmatcher.cc src/PnPsolver.cc src/Sim3Solver.cc src/System.cc src/Tracking.cc src/Viewer.cc
        src/line3D.cc src/view.cc src/clustering.cc src/LineDescriptor.cpp src/EDLineDetector.cpp src/optimization.cc src/sparsematrix.cc src/cudawrapper.cu)
ELSE(L3DPP_CUDA)
        SET(Line3D_SOURCES  src/Converter.cc src/FrameDrawer.cc src/Frame.cc  src/Initializer.cc src/KeyFrame.cc
        src/KeyFrameDatabase.cc src/L3DPPing.cpp src/LocalMapping.cc src/LoopClosing.cc src/Map.cc src/MapPoint.cc src/MapDrawer.cc src/Optimizer.cc
        src/ORBextractor.cc src/ORBmatcher.cc src/PnPsolver.cc src/Sim3Solver.cc src/System.cc src/Tracking.cc src/Viewer.cc 
        src/line3D.cc src/view.cc src/LineDescriptor.cpp src/EDLineDetector.cpp src/optimization.cc src/sparsematrix.cc src/clustering.cc)
ENDIF(L3DPP_CUDA)

# 7 链接库，把刚刚生成的${PROJECT_NAME}库和所需的其它库链接起来
include_directories(${CMAKE_CURRENT_SOURCE_DIR}
${CMAKE_CURRENT_BINARY_DIR}
${CMAKE_CURRENT_SOURCE_DIR}/include)#加入头文件，设置包含的目录

IF(NOT WIN32)
  # shared
  IF(L3DPP_CUDA)
    CUDA_ADD_LIBRARY(${PROJECT_NAME} SHARED ${Line3D_SOURCES} ${Line3D_HEADERS})
  ELSE(L3DPP_CUDA)
    ADD_LIBRARY(${PROJECT_NAME} SHARED ${Line3D_SOURCES} ${Line3D_HEADERS})
  ENDIF(L3DPP_CUDA)
ELSE(NOT WIN32)
  # static
  IF(L3DPP_CUDA)
    CUDA_ADD_LIBRARY(${PROJECT_NAME} ${Line3D_SOURCES} ${Line3D_HEADERS})
  ELSE(L3DPP_CUDA)
    ADD_LIBRARY(${PROJECT_NAME} ${Line3D_SOURCES} ${Line3D_HEADERS})
  ENDIF(L3DPP_CUDA)
ENDIF(NOT WIN32)
target_link_libraries(${PROJECT_NAME} ${ALL_LIBRARIES})

option(APP_LINE_3D++_BUILD_EXECUTABLES "Line3D++: build executables" ON)

IF(APP_LINE_3D++_BUILD_EXECUTABLES)

IF( WIN32 )
  # prevents multiple definition problems in windows via MinGW
  SET(CMAKE_EXE_LINKER_FLAGS "-static")
ENDIF( WIN32 )

# 8 Build examples,编译主函数，生成可执行文件
## 先设置路径
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/Examples/Monocular)
## 可执行文件生成
add_executable(mono_tum
Examples/Monocular/mono_tum.cc)
## 这个可执行文件所需的库（一般就是刚刚生成的工程的库咯）
target_link_libraries(mono_tum ${PROJECT_NAME})

ENDIF(APP_LINE_3D++_BUILD_EXECUTABLES)
