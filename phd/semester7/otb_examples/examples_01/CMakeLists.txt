project(Tutorials)

cmake_minimum_required(VERSION 3.1.0)

find_package(OTB)
if(OTB_FOUND)
  include(${OTB_USE_FILE})
else(OTB_FOUND)
  message(FATAL_ERROR "Cannot build OTB project without OTB. Please set OTB_DIR.")
endif(OTB_FOUND)

add_executable(HelloWorldOTB HelloWorldOTB.cxx )
target_link_libraries(HelloWorldOTB ${OTB_LIBRARIES})

add_executable(Pipeline Pipeline.cxx)
target_link_libraries(Pipeline ${OTB_LIBRARIES})

add_executable(VectorPipeline VectorPipeline.cxx)
target_link_libraries(VectorPipeline ${OTB_LIBRARIES})

add_executable(FilteringPipeline FilteringPipeline.cxx)
target_link_libraries(FilteringPipeline ${OTB_LIBRARIES})

add_executable(ScalingPipeline ScalingPipeline.cxx)
target_link_libraries(ScalingPipeline ${OTB_LIBRARIES})

add_executable(Multispectral Multispectral.cxx)
target_link_libraries(Multispectral ${OTB_LIBRARIES})
