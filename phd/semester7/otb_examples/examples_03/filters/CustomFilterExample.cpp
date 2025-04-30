#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

#include "CustomFilter.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << "<inputImage> <outputImage> <radius>"
              << std::endl;
    return -1;
  }
  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  unsigned int radius = std::stoi(argv[3]);

  // Define Image type
  typedef float PixelType;
  const unsigned int Dimension = 2;
  typedef otb::Image<PixelType, Dimension> ImageType;

  typedef otb::ImageFileReader<ImageType> ReaderType;
  typedef otb::ImageFileWriter<ImageType> WriterType;

  // Instantiate Reader, Custom Filter, and Writer
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  CustomFilter<ImageType>::Pointer customFilter =
      CustomFilter<ImageType>::New();

  reader->SetFileName(inputFileName);
  writer->SetFileName(outputFileName);

  // Set radius for the custom filter
  customFilter->SetRadius(radius);

  // Connect the pipeline
  customFilter->SetInput(reader->GetOutput());
  writer->SetInput(customFilter->GetOutput());

  try {
    writer->Update();
    std::cout << "Custom filter applied and output written to: "
              << outputFileName << std::endl;
  } catch (itk::ExceptionObject &err) {
    std::cerr << "Error: " << err << std::endl;
    return -1;
  }
  return 0;
}
