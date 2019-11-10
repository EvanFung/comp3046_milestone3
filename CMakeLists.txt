cmake_minimum_required(VERSION 3.14)
project(COMP3046_Milestone_3)

set(CMAKE_CXX_STANDARD 14)

#add_executable(COMP3046_Milestone_3 main.cpp)
set(Boost_USE_STATIC_LIBS OFF)
set(Boost_USE_MULTITHREADED ON)
set(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost 1.71.0 COMPONENTS filesystem regex numeric)

if(Boost_FOUND)
    add_executable(COMP3046_Milestone_3 main.cpp)
    target_link_libraries(COMP3046_Milestone_3 ${Boost_LIBRARIES})
endif()