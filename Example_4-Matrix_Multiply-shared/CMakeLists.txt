cmake_minimum_required(VERSION 3.0)

project(MatrixAddS LANGUAGES CXX CUDA) 

find_package(PNG REQUIRED)
find_package(OpenMP REQUIRED)

list(APPEND CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS}")

add_executable(MatrixAddS main.cu utils/pngio.cpp)

target_link_libraries(MatrixAddS ${PNG_LIBRARY})