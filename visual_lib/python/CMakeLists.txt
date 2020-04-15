

### Find packages

find_package(Python3 REQUIRED Interpreter Development)
# find_package(Torch REQUIRED)
message(STATUS "SUBDIR ${pybind11_SUBDIR}")

if(DEFINED pybind11_SUBDIR)
    add_subdirectory(${pybind11_SUBDIR} pybind11)
else()
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


### torc_vilib library

add_library(torch_vilib MODULE
    python/src/torch_vilib.cpp
    python/src/torch_frame.cpp
    python/include/torch_frame.h
)
add_library(vilib::torch ALIAS torch_vilib)
target_include_directories(torch_vilib PUBLIC
    ${TORCH_INCLUDE_DIR}
    "${TORCH_INCLUDE_DIR}/torch/csrc/api/include"
    python/include
)
target_link_libraries(torch_vilib
    PRIVATE
        Python3::Python
        pybind11::module
        vilib::vilib
        Eigen3::Eigen
        #CUDA::cudart
        ${TORCH_LIBRARIES}
)

set_target_properties(torch_vilib PROPERTIES PREFIX "${PYTHON_MODULE_PREFIX}"
                                             SUFFIX "${PYTHON_MODULE_EXTENSION}")
configure_file(python/test.py test.py COPYONLY)
