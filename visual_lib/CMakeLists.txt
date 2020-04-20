cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(vilib VERSION 0.1 LANGUAGES CXX CUDA)

option(WITH_TEST "Build test" OFF)
option(WITH_PYTHON "Build Python bindings" ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)


# Build vilib

aux_source_directory(src SOURCES)
aux_source_directory(src/common SOURCES)
aux_source_directory(src/feature_detection SOURCES)
aux_source_directory(src/feature_detection/fast SOURCES)
aux_source_directory(src/feature_detection/fast/rosten SOURCES)
aux_source_directory(src/feature_tracker SOURCES)
aux_source_directory(src/storage SOURCES)
aux_source_directory(src/preprocess SOURCES)

add_library(vilib ${SOURCES})

add_library(
    vilib::vilib ALIAS vilib
)

set_target_properties(vilib PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)


target_include_directories(vilib
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
)

#target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vilib
    PUBLIC CUDA::cudart Eigen3::Eigen ${OpenCV_LIBS}
)


# Build tests

if(WITH_TEST)
    aux_source_directory(test/src TEST_SOURCES)
    aux_source_directory(test/src/groundtruth TEST_SOURCES)
    aux_source_directory(test/src/feature_detection TEST_SOURCES)
    aux_source_directory(test/src/high_level TEST_SOURCES)
    aux_source_directory(test/src/preprocess TEST_SOURCES)
    aux_source_directory(test/src/storage TEST_SOURCES)


    SET(TEST_NAME test_${CMAKE_PROJECT_NAME})
    add_executable(
        ${TEST_NAME}
        ${TEST_SOURCES}
    )
    target_include_directories(${TEST_NAME} PRIVATE test/include)
    target_link_libraries(${TEST_NAME} vilib ${OpenCV_LIBS})
endif(WITH_TEST)


# Config and install

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    vilibConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(vilibConfig.cmake.in vilibConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vilibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/vilibConfigVersion.cmake"
        DESTINATION lib/cmake/vilib
        )
