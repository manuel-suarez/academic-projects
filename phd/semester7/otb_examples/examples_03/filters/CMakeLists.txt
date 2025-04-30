project(Tutorials)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(CompositeFilterExample CompositeFilterExample.cpp )
target_link_libraries(CompositeFilterExample ${OTB_LIBRARIES})

add_executable(CustomFilterExample CustomFilterExample.cpp )
target_link_libraries(CustomFilterExample ${OTB_LIBRARIES})

add_executable(FunctorImageFilterExample FunctorImageFilterExample.cpp )
target_link_libraries(FunctorImageFilterExample ${OTB_LIBRARIES})

add_executable(NeighborhoodIteratorFilterExample NeighborhoodIteratorFilterExample.cpp )
target_link_libraries(NeighborhoodIteratorFilterExample ${OTB_LIBRARIES})
