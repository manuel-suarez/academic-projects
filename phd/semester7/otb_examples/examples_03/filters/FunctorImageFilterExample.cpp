#include "itkUnaryFunctorImageFilter.h"
#include "otbImage.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include <cmath>

// Define the functor for a logarithmic transformation
namespace itk {
namespace Functor {
template <typename TInputType, typename TOutputType> class LogTransform {
public:
  LogTransform() : m_Scale(1.0) {}

  // Set/Get the scale factor
  void SetScale(double scale) { m_Scale = scale; }
  double GetScale() const { return m_Scale; }

  // Pixel-wise operation
  TOutputType operator()(const TInputType &input) const {
    // Ensure no negative or zero input for log operation
    return static_cast<TOutputType>(
        std::log(1.0 + m_Scale * static_cast<double>(input)));
  }

  bool operator==(const LogTransform &other) const { return true; }

  bool operator!=(const LogTransform &other) const { return !(*this == other); }

private:
  double m_Scale;
};

} // namespace Functor
} // namespace itk
  //
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <inputImage> <outputImage> <scaleFactor>" << std::endl;
    return -1;
  }

  const char *inputFileName = argv[1];
  const char *outputFileName = argv[2];
  const double scaleFactor = std::stod(argv[3]);

  // Define image types
  constexpr unsigned int Dimension = 2;
  typedef float PixelType; // Float for SAR image precision
  typedef otb::Image<PixelType, Dimension> ImageType;

  // Define the reader and writer
  typedef otb::ImageFileReader<ImageType> ReaderType;
  typedef otb::ImageFileWriter<ImageType> WriterType;

  // Define the functor filter type
  typedef itk::Functor::LogTransform<PixelType, PixelType> FunctorType;
  typedef itk::UnaryFunctorImageFilter<ImageType, ImageType, FunctorType>
      FilterType;

  // Instantiate the pipeline components
  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  FilterType::Pointer filter = FilterType::New();

  // Set file names
  reader->SetFileName(inputFileName);
  writer->SetFileName(outputFileName);

  // Configure the functor
  FunctorType functor;
  functor.SetScale(scaleFactor);

  // Connect the filter and the functor
  filter->SetFunctor(functor);
  filter->SetInput(reader->GetOutput());
  writer->SetInput(filter->GetOutput());

  // Execute the pipeline
  try

  {
    writer->Update();
    std::cout << "Logarithmic transformation applied with scale factor: "
              << scaleFactor << std::endl;
  } catch (itk::ExceptionObject &err) {
    std::cerr << "Error: " << err << std::endl;
    return -1;
  }

  return 0;
}
