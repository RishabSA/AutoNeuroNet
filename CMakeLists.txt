# cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
# cmake --build build

cmake_minimum_required(VERSION 3.20)
project(AutoNeuroNet)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
    OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(pybind11_DIR ${PYBIND11_CMAKE_DIR})
find_package(pybind11 REQUIRED)

pybind11_add_module(_autoneuronet
    pybind_wrapper.cpp
    src/Var.cpp
    src/Matrix.cpp
    src/NeuralNetwork.cpp
    src/Optimizers.cpp
    src/LossFunctions.cpp
)

target_include_directories(_autoneuronet PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

install(TARGETS _autoneuronet
    LIBRARY DESTINATION autoneuronet
    ARCHIVE DESTINATION autoneuronet
    RUNTIME DESTINATION autoneuronet
)
