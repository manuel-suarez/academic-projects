#include "itkMeanImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <inputImage> <outputImage>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];

  typedef unsigned char PixelType;
  const unsigned int Dimension = 2;
  typedef otb::Image<PixelType, Dimension> ImageType;

  typedef otb::ImageFileReader<ImageType> ReaderType;
  typedef otb::ImageFileWriter<ImageType> WriterType;

  typedef itk::MeanImageFilter<ImageType, ImageType> MeanFilterType;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  MeanFilterType::Pointer meanFilter = MeanFilterType::New();

  reader->SetFileName(inputFileName);
  writer->SetFileName(outputFileName);

  // Set up the filter radius
  MeanFilterType::InputSizeType radius;
  radius.Fill(3); // 3x3 neighborhood
  meanFilter->SetRadius(radius);

  // Connect the pipeline: Reader → Filter → Writer
  meanFilter->SetInput(reader->GetOutput());
  writer->SetInput(meanFilter->GetOutput());

  try {
    writer->Update();
    std::cout << "Image successfully filtered and written to: "
              << outputFileName << std::endl;
  } catch (itk::ExceptionObject &err) {
    std::cerr << "Error: " << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
