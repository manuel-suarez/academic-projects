#include "itkGradientMagnitudeImageFilter.h"
#include "itkRescaleIntensityImageFilter.h"
#include "itkThresholdImageFilter.h"
#include "itkUnaryFunctorImageFilter.h"

#include "itkNumericTraits.h"
#include "otbImage.h"

namespace otb {

template <class TImageType>
class ITK_EXPORT CompositeExampleImageFilter
    : public itk::ImageToImageFilter<TImageType, TImageType> {
public:
  // Standard declarations, used for object creation with the object factory
  using Self = CompositeExampleImageFilter<TImageType>;
  using Superclass = itk::ImageToImageFilter<TImageType, TImageType>;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  /** Method for creation through object factory */
  itkNewMacro(Self);

  /** Run-time type information */
  itkTypeMacro(CompositeExampleImageFilter, itk::ImageToImageFilter);

  /** Display */
  void PrintSelf(std::ostream &os, itk::Indent indent) const override;

  using PixelType = typename TImageType::PixelType;

  itkGetMacro(Threshold, PixelType);
  itkSetMacro(Threshold, PixelType);

protected:
  CompositeExampleImageFilter();

protected:
  using ThresholdType = itk::ThresholdImageFilter<TImageType>;
  using GradientType =
      itk::GradientMagnitudeImageFilter<TImageType, TImageType>;
  using RescalerType = itk::RescaleIntensityImageFilter<TImageType, TImageType>;

  void GenerateData() override;

private:
  CompositeExampleImageFilter(Self &);
  void operator=(const Self &);

  typename GradientType::Pointer m_GradientFilter;
  typename ThresholdType::Pointer m_ThresholdFilter;
  typename RescalerType::Pointer m_RescaleFilter;

  PixelType m_Threshold;
};

} // namespace otb

namespace otb {
template <class TImageType>
CompositeExampleImageFilter<TImageType>::CompositeExampleImageFilter() {
  m_GradientFilter = GradientType::New();
  m_ThresholdFilter = ThresholdType::New();
  m_RescaleFilter = RescalerType::New();

  m_ThresholdFilter->SetInput(m_GradientFilter->GetOutput());
  m_RescaleFilter->SetInput(m_ThresholdFilter->GetOutput());

  m_Threshold = 1;

  m_RescaleFilter->SetOutputMinimum(
      itk::NumericTraits<PixelType>::NonpositiveMin());
  m_RescaleFilter->SetOutputMaximum(itk::NumericTraits<PixelType>::max());
}

template <class TImageType>
void CompositeExampleImageFilter<TImageType>::GenerateData() {
  m_GradientFilter->SetInput(this->GetInput());
  m_ThresholdFilter->ThresholdBelow(this->m_Threshold);

  m_RescaleFilter->GraftOutput(this->GetOutput());
  m_RescaleFilter->Update();

  this->GraftOutput(m_RescaleFilter->GetOutput());
}

template <class TImageType>
void CompositeExampleImageFilter<TImageType>::PrintSelf(
    std::ostream &os, itk::Indent indent) const {
  Superclass::PrintSelf(os, indent);

  os << indent << "Threshold: " << this->m_Threshold << std::endl;
}

} // namespace otb

#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"

int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << std::endl;
    std::cerr << argv[0] << " inputImageFile outputImageFile" << std::endl;
    return -1;
  }

  using ImageType = otb::Image<short, 2>;
  using ReaderType = otb::ImageFileReader<ImageType>;
  using WriterType = otb::ImageFileWriter<ImageType>;

  using FilterType = otb::CompositeExampleImageFilter<ImageType>;

  ReaderType::Pointer reader = ReaderType::New();
  WriterType::Pointer writer = WriterType::New();
  FilterType::Pointer filter = FilterType::New();

  reader->SetFileName(argv[1]);
  filter->SetInput(reader->GetOutput());
  filter->SetThreshold(20);
  writer->SetInput(filter->GetOutput());
  writer->SetFileName(argv[2]);

  try {
    writer->Update();
  } catch (itk::ExceptionObject &e) {
    std::cerr << "Error: " << e << std::endl;
  }

  return 0;
}
