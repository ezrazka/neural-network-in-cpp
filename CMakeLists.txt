cmake_minimum_required(VERSION 3.20)
project(neuralnetwork LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_library(nn_lib
    src/vector.cpp
    src/matrix.cpp
)
target_include_directories(nn_lib PUBLIC include)

add_executable(neuralnetwork src/main.cpp)
target_link_libraries(neuralnetwork PRIVATE nn_lib)
