#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

#include "itkNeighborhoodInnerProduct.h"
#include "itkSobelOperator.h"

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Missing parameters. " << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile direction"
              << std::endl;
    return -1;
  }

  using PixelType = float;
  using ImageType = otb::Image<PixelType, 2>;
  using ReaderType = otb::ImageFileReader<ImageType>;

  using NeighborhoodIteratorType = itk::ConstNeighborhoodIterator<ImageType>;
  using IteratorType = itk::ImageRegionIterator<ImageType>;

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &err) {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
  }

  ImageType::Pointer output = ImageType::New();
  output->SetRegions(reader->GetOutput()->GetRequestedRegion());
  output->Allocate();

  IteratorType out(output, reader->GetOutput()->GetRequestedRegion());

  itk::SobelOperator<PixelType, 2> sobelOperator;
  sobelOperator.SetDirection(::atoi(argv[3]));
  sobelOperator.CreateDirectional();

  NeighborhoodIteratorType::RadiusType radius = sobelOperator.GetRadius();
  NeighborhoodIteratorType it(radius, reader->GetOutput(),
                              reader->GetOutput()->GetRequestedRegion());

  itk::NeighborhoodInnerProduct<ImageType> innerProduct;

  for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out) {
    out.Set(innerProduct(it, sobelOperator));
  }

  using WritePixelType = unsigned char;
  using WriteImageType = otb::Image<WritePixelType, 2>;
  using WriterType = otb::ImageFileWriter<WriteImageType>;

  using RescaleFilterType =
      itk::RescaleIntensityImageFilter<ImageType, WriteImageType>;

  RescaleFilterType::Pointer rescaler = RescaleFilterType::New();

  rescaler->SetOutputMinimum(0);
  rescaler->SetOutputMaximum(255);
  rescaler->SetInput(output);

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[2]);
  writer->SetInput(rescaler->GetOutput());
  try {
    writer->Update();
  } catch (itk::ExceptionObject &err) {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
  }

  return 0;
}
