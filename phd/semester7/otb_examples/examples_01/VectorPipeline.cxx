#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include "otbVectorImage.h"
#include <cstdlib>
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename>"
              << std::endl;
  }

  using ImageType = otb::VectorImage<unsigned char, 2>;

  using ReaderType = otb::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();

  using WriterType = otb::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName(argv[1]);
  writer->SetFileName(argv[2]);

  writer->SetInput(reader->GetOutput());
  writer->Update();

  return EXIT_SUCCESS;
}
