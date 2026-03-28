cmake_minimum_required(VERSION 3.20)
project(neuralnetwork LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -funroll-loops -flto -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "-fsanitize=address,undefined")

add_library(nn_lib
    src/vector.cpp
    src/matrix.cpp
)
target_include_directories(nn_lib PUBLIC include)

add_executable(neuralnetwork src/main.cpp)
target_link_libraries(neuralnetwork PRIVATE nn_lib)
