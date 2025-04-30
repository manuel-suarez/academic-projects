project(OTB_Iterators)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(NeighborhoodIterators1 NeighborhoodIterators1.cxx)
target_link_libraries(NeighborhoodIterators1 ${OTB_LIBRARIES})

add_executable(NeighborhoodIterators1a NeighborhoodIterators1a.cxx)
target_link_libraries(NeighborhoodIterators1a ${OTB_LIBRARIES})

add_executable(NeighborhoodIteratorsMean NeighborhoodIteratorsMean.cxx)
target_link_libraries(NeighborhoodIteratorsMean ${OTB_LIBRARIES})

add_executable(NeighborhoodIteratorsVariance NeighborhoodIteratorsVariance.cxx)
target_link_libraries(NeighborhoodIteratorsVariance ${OTB_LIBRARIES})

