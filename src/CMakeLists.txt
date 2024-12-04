project(faldoi_qtcreator)
cmake_minimum_required(VERSION 2.8.1)
aux_source_directory(. SRC_LIST)

message("Build type: ${CMAKE_BUILD_TYPE}")

# Find OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find OpenMP
find_package(OpenMP)

# Set compiler flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -std=c99 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -std=c++11")

set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

if (CMAKE_CXX_COMPILER EQUAL clang++)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvectorize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvectorize")
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ftree-vectorize -ftree-loop-vectorize")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftree-vectorize -ftree-loop-vectorize")
endif()

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS} -O3")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS} -O0 -ggdb -DNDEBUG -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -ggdb -DNDEBUG -Wall -Wextra")

# Shared source files
SET(SHARED_C_SRC iio.c mask.c xmalloc.c bicubic_interpolation.c elap_recsep.c)
SET(SHARED_CPP_SRC 
    tvl2_model.cpp nltv_model.cpp tvcsad_model.cpp nltvcsad_model.cpp 
    tvl2w_model.cpp nltvcsadw_model.cpp nltvw_model.cpp tvcsadw_model.cpp 
    aux_energy_model.cpp energy_model.cpp tvl2_model_occ.cpp utils.cpp 
    utils_preprocess.cpp aux_partitions.cpp)

# Video denoising source files
SET(VIDEO_DENOISING_SRC
    VideoIO.cpp
    MotionDenoiser.cpp
    main.cpp)

# Build original FALDOI executables
add_executable(sparse_flow ${SHARED_C_SRC} ${SHARED_CPP_SRC} sparse_flow.cpp)
add_executable(local_faldoi ${SHARED_C_SRC} ${SHARED_CPP_SRC} local_faldoi.cpp)
add_executable(global_faldoi ${SHARED_C_SRC} ${SHARED_CPP_SRC} global_faldoi.cpp)

# Build video denoising executable
add_executable(video_denoiser ${SHARED_C_SRC} ${SHARED_CPP_SRC} ${VIDEO_DENOISING_SRC})

# Link libraries for FALDOI executables
target_link_libraries(sparse_flow 
    ${OpenCV_LIBS}  # OpenCV libraries
    -lz png jpeg tiff)

target_link_libraries(local_faldoi 
    ${OpenCV_LIBS}  # OpenCV libraries
    -lz png jpeg tiff)

target_link_libraries(global_faldoi 
    ${OpenCV_LIBS}  # OpenCV libraries
    -lz png jpeg tiff)

# Link libraries for video denoising executable
target_link_libraries(video_denoiser 
    ${OpenCV_LIBS}  # OpenCV libraries
    -lz png jpeg tiff)

# Print OpenCV information for debugging
message(STATUS "OpenCV_INCLUDE_DIRS = ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS = ${OpenCV_LIBS}")


