#include "itkCannyEdgeDetectionImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 3) {
    std::cerr << "Usage: " << argv[0] << " <input_filename> <output_filename>"
              << std::endl;
  }

  using PixelType = double;
  using ImageType = otb::Image<PixelType, 2>;

  using OutputPixelType = unsigned char;
  using OutputImageType = otb::Image<OutputPixelType, 2>;

  using ReaderType = otb::ImageFileReader<ImageType>;
  ReaderType::Pointer reader = ReaderType::New();

  using WriterType = otb::ImageFileWriter<OutputImageType>;
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName(argv[1]);
  writer->SetFileName(argv[2]);

  using FilterType = itk::CannyEdgeDetectionImageFilter<ImageType, ImageType>;
  FilterType::Pointer filter = FilterType::New();

  using RescalerType =
      itk::RescaleIntensityImageFilter<ImageType, OutputImageType>;
  RescalerType::Pointer rescaler = RescalerType::New();

  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(255);

  // Pipeline
  filter->SetInput(reader->GetOutput());
  rescaler->SetInput(filter->GetOutput());
  writer->SetInput(rescaler->GetOutput());

  writer->Update();

  return EXIT_SUCCESS;
}
