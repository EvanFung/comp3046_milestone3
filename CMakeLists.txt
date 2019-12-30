cmake_minimum_required(VERSION 3.14.5)
project(COMP3046_Milestone_3)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(annTraining src/main.cpp src/NeuralNetwork.cpp)
cmake_minimum_required(VERSION 2.8.8 FATAL_ERROR)