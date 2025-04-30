#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include "otbLocalStatisticExtractionFilter.h"

int main(int argc, char *argv[]) {
  typedef otb::Image<float, 2> ImageType;

  if (argc < 4) {
    std::cerr << "Usage: " << argv[0] << " <inputImage> <outputImage> <radius>"
              << std::endl;
    return EXIT_FAILURE;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  unsigned int radius = std::stoi(argv[3]);

  auto reader = otb::ImageFileReader<ImageType>::New();
  reader->SetFileName(inputFileName);

  typedef otb::LocalStatisticExtractionFilter<ImageType, ImageType>
      VarianceFilterType;
  auto varianceFilter = VarianceFilterType::New();
  varianceFilter->SetRadius(radius);
  varianceFilter->SetInput(reader->GetOutput());
  varianceFilter->SetComputeVariance(true);

  auto writer = otb::ImageFileWriter<ImageType>::New();
  writer->SetFileName(outputFileName);
  writer->SetInput(varianceFilter->GetOutput());

  writer->Update();

  return EXIT_SUCCESS;
}
