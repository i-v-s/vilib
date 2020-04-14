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

add_library(
    vilib SHARED
    ${SOURCES}
)

add_library(
    vilib::vilib ALIAS vilib
)

set_target_properties(vilib PROPERTIES
    CXX_STANDARD 17
    CXX_EXTENSIONS OFF
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


# Build Python bindings

if(WITH_PYTHON)
    find_package(Python3 REQUIRED Interpreter Development)
    # find_package(Torch REQUIRED)
    if(NOT DEFINED PYBIND11_VERSION_MAJOR)
        find_package(pybind11 REQUIRED)
    endif()

    #add_subdirectory(python/pybind11)
    #pybind11_add_module(pyvilib python/src/vilib.cpp)

    message(STATUS "Python3_EXECUTABLE is ${Python3_EXECUTABLE}")
    exec_program(${Python3_EXECUTABLE}
                 ARGS "-c \"from os.path import join, split; import torch; print(split(torch.__file__)[0])\""
                 OUTPUT_VARIABLE TORCH_DIR
                 RETURN_VALUE TORCH_NOT_FOUND
                )
    set(TORCH_INCLUDE_DIR ${TORCH_DIR}/include)
    set(TORCH_LIB_DIR ${TORCH_DIR}/lib)
    set(TORCH_LIBRARIES
        ${TORCH_LIB_DIR}/libtorch.so
        ${TORCH_LIB_DIR}/libc10.so
        ${TORCH_LIB_DIR}/libc10_cuda.so
        ${TORCH_LIB_DIR}/libtorch_python.so
    )

    message(STATUS "TORCH_INCLUDE_DIR is ${TORCH_INCLUDE_DIR}")
    message(STATUS "TORCH_LIBRARIES is ${TORCH_LIBRARIES}")
    add_library(pyvilib MODULE
        python/src/vilib.cpp
        python/src/torch_frame.cpp
        python/include/torch_frame.h
    )
    target_include_directories(pyvilib PRIVATE
        ${TORCH_INCLUDE_DIR}
        "${TORCH_INCLUDE_DIR}/torch/csrc/api/include"
        python/include
    )
    target_link_libraries(pyvilib PRIVATE
        Python3::Python
        pybind11::module
        vilib
        Eigen3::Eigen
        #CUDA::cudart
        ${TORCH_LIBRARIES}
    )
    set_target_properties(pyvilib PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
    configure_file(python/test.py test.py COPYONLY)
endif(WITH_PYTHON)


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

install(
    TARGETS vilib
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    vilibConfigVersion.cmake
    VERSION ${PACKAGE_VERSION}
    COMPATIBILITY AnyNewerVersion
)

export(TARGETS vilib NAMESPACE vilib:: FILE vilibTargets.cmake)

configure_file(vilibConfig.cmake.in vilibConfig.cmake @ONLY)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/vilibConfig.cmake"
              "${CMAKE_CURRENT_BINARY_DIR}/vilibConfigVersion.cmake"
        DESTINATION lib/cmake/vilib
        )
