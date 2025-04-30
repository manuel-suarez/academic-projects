#include "itkImageRegionIterator.h"
#include "itkNeighborhoodIterator.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

constexpr unsigned int Dimension = 2;
using PixelType = float;
using ImageType = otb::Image<PixelType, Dimension>;

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <inputImage> <outputImage> [radius]"
              << std::endl;
    return -1;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  const unsigned int radius = (argc > 3) ? std::stoi(argv[3]) : 1;

  // 1. Create the reader and writer
  using ReaderType = otb::ImageFileReader<ImageType>;
  using WriterType = otb::ImageFileWriter<ImageType>;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();

  reader->SetFileName(inputFileName);
  writer->SetFileName(outputFileName);

  // 2. Read the input image
  reader->Update();
  ImageType::Pointer inputImage = reader->GetOutput();

  // 3. Create an output image to store the results
  ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
  outputImage->Allocate();
  outputImage->FillBuffer(0);

  // 4. Set up the NeighborhoodIterator
  itk::Size<Dimension> neighborhoodRadius;
  neighborhoodRadius.Fill(radius);

  itk::NeighborhoodIterator<ImageType> neighborhoodIt(
      neighborhoodRadius, inputImage, inputImage->GetRequestedRegion());
  itk::ImageRegionIterator<ImageType> outputIt(
      outputImage, outputImage->GetRequestedRegion());

  // 5. Iterate tthrough the image and compute the mean in the neighborhood
  for (neighborhoodIt.GoToBegin(), outputIt.GoToBegin();
       !neighborhoodIt.IsAtEnd(); ++neighborhoodIt, ++outputIt) {
    double sum = 0.0;
    unsigned int count = 0;

    // Iterate through the neighborhood
    for (unsigned int i = 0; i < neighborhoodIt.Size(); ++i) {
      if (neighborhoodIt.GetPixel(i) > 0) {
        sum += neighborhoodIt.GetPixel(i);
        ++count;
      }
    }

    outputIt.Set(static_cast<PixelType>(sum / count));
  }

  // 6. Write the output image
  writer->SetInput(outputImage);
  try {
    writer->Update();
    std::cout << "Neightborhood mean filter applied successfully." << std::endl;
  } catch (itk::ExceptionObject &err) {
    std::cerr << "Error: " << err << std::endl;
    return -1;
  }

  return 0;
}
