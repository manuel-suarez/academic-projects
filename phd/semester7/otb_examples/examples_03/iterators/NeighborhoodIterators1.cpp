#include "itkRescaleIntensityImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

#include "itkConstNeighborhoodIterator.h"
#include "itkImageRegionIterator.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Missing parameters. " << std::endl;
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile" << std::endl;
    return -1;
  }

  // The finite difference calculations
  // in this algorithm require floating point values.  Hence, we define the
  // image pixel type to be \code{float} and the file reader will automatically
  // cast fixed-point data to \code{float}.
  //
  // We declare the iterator types using the image type as
  // the template parameter. The second template parameter of the
  // neighborhood iterator, which specifies
  // the boundary condition, has been omitted because the default condition is
  // appropriate for this algorithm.

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
    std::cout << "ExceptionObject caught!" << std::endl;
    std::cout << err << std::endl;
    return -1;
  }

  NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(1);
  NeighborhoodIteratorType it(radius, reader->GetOutput(),
                              reader->GetOutput()->GetRequestedRegion());

  ImageType::Pointer output = ImageType::New();
  output->SetRegions(reader->GetOutput()->GetRequestedRegion());
  output->Allocate();

  IteratorType out(output, reader->GetOutput()->GetRequestedRegion());
  NeighborhoodIteratorType::OffsetType offset1 = {{-1, -1}};
  NeighborhoodIteratorType::OffsetType offset2 = {{1, -1}};
  NeighborhoodIteratorType::OffsetType offset3 = {{-1, 0}};
  NeighborhoodIteratorType::OffsetType offset4 = {{1, 0}};
  NeighborhoodIteratorType::OffsetType offset5 = {{-1, 1}};
  NeighborhoodIteratorType::OffsetType offset6 = {{1, 1}};

  for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out) {
    float sum;
    sum = it.GetPixel(offset2) - it.GetPixel(offset1);
    sum += 2.0 * it.GetPixel(offset4) - 2.0 * it.GetPixel(offset3);
    sum += it.GetPixel(offset6) - it.GetPixel(offset5);
    out.Set(sum);
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
