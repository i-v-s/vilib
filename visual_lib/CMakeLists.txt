cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(vilib LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Build vilib

aux_source_directory(src SOURCES)
aux_source_directory(src/common SOURCES_COMMON)
aux_source_directory(src/feature_detection SOURCES_FEATURE_DETECTION)
aux_source_directory(src/feature_detection/fast SOURCES_FEATURE_DETECTION)
aux_source_directory(src/feature_detection/fast/rosten SOURCES_FEATURE_DETECTION)
aux_source_directory(src/feature_tracker SOURCES_FEATURE_TRACKER)
aux_source_directory(src/storage SOURCES_STORAGE)
aux_source_directory(src/preprocess SOURCES_PREPROCESS)

add_library(
    ${CMAKE_PROJECT_NAME} SHARED
    ${SOURCES}
    ${SOURCES_COMMON}
    ${SOURCES_FEATURE_DETECTION}
    ${SOURCES_FEATURE_TRACKER}
    ${SOURCES_STORAGE}
    ${SOURCES_PREPROCESS}
)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
    CXX_STANDARD 11
    CXX_EXTENSIONS OFF
)
target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(${CMAKE_PROJECT_NAME} CUDA::cudart Eigen3::Eigen ${OpenCV_LIBS})


# Build tests

aux_source_directory(test/src TEST_SOURCES)
aux_source_directory(test/src/groundtruth TEST_SOURCES_GT)
aux_source_directory(test/src/feature_detection TEST_SOURCES_FEATURE_DETECTION)
aux_source_directory(test/src/high_level TEST_SOURCES_HL)
aux_source_directory(test/src/preprocess TEST_SOURCES_PREPROCESS)
aux_source_directory(test/src/storage TEST_SOURCES_STORAGE)


SET(TEST_NAME test_${CMAKE_PROJECT_NAME})
add_executable(
    ${TEST_NAME}
    ${TEST_SOURCES}
    ${TEST_SOURCES_GT}
    ${TEST_SOURCES_HL}
    ${TEST_SOURCES_FEATURE_DETECTION}
    ${TEST_SOURCES_PREPROCESS}
    ${TEST_SOURCES_STORAGE}
)
target_include_directories(${TEST_NAME} PRIVATE test/include)
target_link_libraries(${TEST_NAME} ${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
