#include "itkGradientMagnitudeImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename>"
              << std::endl;
  }

  using ImageType = otb::Image<unsigned char, 2>;
  using ReaderType = otb::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();

  using WriterType = otb::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName(argv[1]);
  writer->SetFileName(argv[2]);

  using FilterType = itk::GradientMagnitudeImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();

  filter->SetInput(reader->GetOutput());
  writer->SetInput(filter->GetOutput());

  writer->Update();

  return EXIT_SUCCESS;
}
