#include "otbLeeImageFilter.h"

#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " inputImageFile ";
    std::cerr << " outputImageFile radius NbLooks" << std::endl;
    return EXIT_FAILURE;
  }

  using PixelType = double;

  // The images are defined using the pixel type and the dimension.
  using InputImageType = otb::Image<PixelType, 2>;
  using OutputImageType = otb::Image<PixelType, 2>;

  // The filter can be instantiated using the image types defined above.
  using FilterType = otb::LeeImageFilter<InputImageType, OutputImageType>;

  // An ImageFileReader class is also instantiated in order to read
  // image data from a file.
  using ReaderType = otb::ImageFileReader<InputImageType>;

  // An ImageFileWriter is instantiated in order to write the
  // output image to a file.
  using WriterType = otb::ImageFileWriter<OutputImageType>;

  // Both the filter and the reader are created by invoking their New()
  // methods and assigning the result to SmartPointers.
  ReaderType::Pointer reader = ReaderType::New();
  FilterType::Pointer filter = FilterType::New();

  WriterType::Pointer writer = WriterType::New();
  writer->SetInput(filter->GetOutput());
  reader->SetFileName(argv[1]);

  // The image obtained with the reader is passed as input to the
  // LeeImageFilter.
  filter->SetInput(reader->GetOutput());

  // The method SetRadius() defines the size of the window to
  // be used for the computation of the local statistics. The method
  // SetNbLooks() sets the number of looks of the input
  // image.
  FilterType::SizeType Radius;
  Radius[0] = atoi(argv[3]);
  Radius[1] = atoi(argv[3]);

  filter->SetRadius(Radius);
  filter->SetNbLooks(atoi(argv[4]));

  writer->SetFileName(argv[2]);
  writer->Update();
}
