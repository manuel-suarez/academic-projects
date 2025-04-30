project(Tutorials)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(ImageRegionIterator ImageRegionIterator.cpp )
target_link_libraries(ImageRegionIterator ${OTB_LIBRARIES})

add_executable(NeighborhoodIterators1 NeighborhoodIterators1.cpp)
target_link_libraries(NeighborhoodIterators1 ${OTB_LIBRARIES})

add_executable(NeighborhoodIterators2 NeighborhoodIterators2.cpp)
target_link_libraries(NeighborhoodIterators2 ${OTB_LIBRARIES})
