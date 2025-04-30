#include "itkImageRegionIterator.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include <iostream>

int main(int argc, char *argv[]) {
  if (argc < 7) {
    std::cerr << "Missing parameters. " << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0]
              << " inputImageFile outputImageFile startX startY sizeX sizeY"
              << std::endl;
    return -1;
  }

  const unsigned int Dimension = 2;

  using PixelType = double;
  using ImageType = otb::Image<PixelType, Dimension>;

  using ConstIteratorType = itk::ImageRegionConstIterator<ImageType>;
  using IteratorType = itk::ImageRegionIterator<ImageType>;

  using ReaderType = otb::ImageFileReader<ImageType>;
  using WriterType = otb::ImageFileWriter<ImageType>;

  ImageType::RegionType inputRegion;
  ImageType::RegionType::IndexType inputStart;
  ImageType::RegionType::SizeType size;

  inputStart[0] = ::atoi(argv[3]);
  inputStart[1] = ::atoi(argv[4]);

  size[0] = ::atoi(argv[5]);
  size[1] = ::atoi(argv[6]);

  inputRegion.SetSize(size);
  inputRegion.SetIndex(inputStart);

  ImageType::RegionType outputRegion;
  ImageType::RegionType::IndexType outputStart;

  outputStart[0] = 0;
  outputStart[1] = 0;

  outputRegion.SetSize(size);
  outputRegion.SetIndex(outputStart);

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &err) {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return -1;
  }

  if (!reader->GetOutput()->GetRequestedRegion().IsInside(inputRegion)) {
    std::cerr << "Error" << std::endl;
    std::cerr << "The region " << inputRegion
              << "is not contained within the input image region " << std::endl;
    return -1;
  }

  ImageType::Pointer outputImage = ImageType::New();
  outputImage->SetRegions(outputRegion);
  const ImageType::SpacingType &spacing =
      reader->GetOutput()->GetSignedSpacing();
  const ImageType::PointType &inputOrigin = reader->GetOutput()->GetOrigin();
  double outputOrigin[Dimension];

  for (unsigned int i = 0; i < Dimension; i++) {
    outputOrigin[i] = inputOrigin[i] + spacing[i] * inputStart[i];
  }

  outputImage->SetSignedSpacing(spacing);
  outputImage->SetOrigin(outputOrigin);
  outputImage->Allocate();

  ConstIteratorType inputIt(reader->GetOutput(), inputRegion);
  IteratorType outputIt(outputImage, outputRegion);

  for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd();
       ++inputIt, ++outputIt) {
    outputIt.Set(inputIt.Get());
  }

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[2]);
  writer->SetInput(outputImage);

  try {
    writer->Update();
  } catch (itk::ExceptionObject &err) {
    std::cerr << "ExceptionObject caught!" << std::endl;
    std::cerr << err << std::endl;
    return -1;
  }

  return 0;
}
