project(Variance)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(VarianceFilter VarianceFilter.cxx )
target_link_libraries(VarianceFilter ${OTB_LIBRARIES})
