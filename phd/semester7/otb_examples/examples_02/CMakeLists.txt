project(BasicOTBExample)
cmake_minimum_required(VERSION 3.10)

find_package(OTB REQUIRED)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(ImageExample ImageExample.cpp)
target_link_libraries(ImageExample ${OTB_LIBRARIES})

add_executable(ImageFilter ImageFilter.cpp)
target_link_libraries(ImageFilter ${OTB_LIBRARIES})
