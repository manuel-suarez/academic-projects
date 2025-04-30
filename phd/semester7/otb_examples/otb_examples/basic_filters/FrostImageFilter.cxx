#include "otbFrostImageFilter.h"

#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

int main(int argc, char *argv[]) {

  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " inputImageFile ";
    std::cerr << " outputImageFile radius deramp" << std::endl;
    return EXIT_FAILURE;
  }

  using PixelType = unsigned char;
  using InputImageType = otb::Image<PixelType, 2>;
  using OutputImageType = otb::Image<PixelType, 2>;

  // The filter can be instantiated using the image types defined previously.
  using FilterType = otb::FrostImageFilter<InputImageType, OutputImageType>;
  using ReaderType = otb::ImageFileReader<InputImageType>;
  using WriterType = otb::ImageFileWriter<OutputImageType>;

  ReaderType::Pointer reader = ReaderType::New();
  FilterType::Pointer filter = FilterType::New();

  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(filter->GetOutput());
  reader->SetFileName(argv[1]);

  // The image obtained with the reader is passed as input to the
  // FrostImageFilter
  filter->SetInput(reader->GetOutput());

  // The method SetRadius() defines the size of the window to
  // be used for the computation of the local statistics. The method
  // SetDeramp() sets the K coefficient.
  FilterType::SizeType Radius;
  Radius[0] = atoi(argv[3]);
  Radius[1] = atoi(argv[3]);

  filter->SetRadius(Radius);
  filter->SetDeramp(atof(argv[4]));

  writer->SetFileName(argv[2]);
  writer->Update();
}
