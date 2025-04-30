cmake_minimum_required(VERSION 3.23)
project(gaussian_blur_opencv)
find_package(OpenCV REQUIRED)
set(CMAKE_CXX_STANDARD 14)

add_executable(gaussian_blur_opencv main.cpp)
target_link_libraries(gaussian_blur_opencv ${OpenCV_LIBS})
