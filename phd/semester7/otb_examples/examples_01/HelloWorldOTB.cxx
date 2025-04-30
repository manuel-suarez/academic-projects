// The following code is an implementation of a small OTB
// program. It tests including header files and linking with OTB
// libraries

#include "otbImage.h"
#include <cstdlib>
#include <iostream>

int main(int itkNotUsed(argc), char *itkNotUsed(argv)[]) {
  using ImageType = otb::Image<unsigned short, 2>;

  ImageType::Pointer image = ImageType::New();

  std::cout << "OTB Hello World!" << std::endl;

  return EXIT_SUCCESS;
}
