#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

int main(int argc, char *argv[]) {
  // Check for input arguments: input file and output file
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <inputImage> <outputImage>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];

  // Define Image type: Pixel Type, Dimension
  typedef unsigned char PixelType;
  const unsigned int Dimension = 2;
  typedef otb::Image<PixelType, Dimension> ImageType;

  // Define Reader and Writer types
  typedef otb::ImageFileReader<ImageType> ReaderType;
  typedef otb::ImageFileWriter<ImageType> WriterType;

  // Instantiate Reader and Writer
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  // Set input and output file names
  reader->SetFileName(inputFileName);
  writer->SetFileName(outputFileName);

  // Connect pipeline: Reader → Writer
  writer->SetInput(reader->GetOutput());

  // Execute pipeline
  try {
    writer->Update();
    std::cout << "Image successfully read and written to: " << outputFileName
              << std::endl;
  } catch (itk::ExceptionObject &err) {
    std::cerr << "Error: " << err << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
