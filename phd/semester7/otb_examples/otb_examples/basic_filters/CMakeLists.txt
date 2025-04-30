project(OTB_BasicFilters)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(BandMathFilterExample BandMathFilterExample.cxx)
target_link_libraries(BandMathFilterExample ${OTB_LIBRARIES})

add_executable(FrostImageFilter FrostImageFilter.cxx)
target_link_libraries(FrostImageFilter ${OTB_LIBRARIES})

add_executable(LeeImageFilter LeeImageFilter.cxx)
target_link_libraries(LeeImageFilter ${OTB_LIBRARIES})
