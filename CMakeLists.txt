cmake_minimum_required(VERSION 4.1)
project(automatic_differentiation)

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

pybind11_add_module(Var
  pybind_wrapper.cpp
  src/Var.cpp
)

target_include_directories(Var PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)