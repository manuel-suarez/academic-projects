#include "itkImageRegionIterator.h"
#include "itkImageToImageFilter.h"
#include "otbImage.h"

template <typename TImageType>
class CustomFilter : public itk::ImageToImageFilter<TImageType, TImageType> {
public:
  typedef CustomFilter Self;
  typedef itk::ImageToImageFilter<TImageType, TImageType> Superclass;
  typedef itk::SmartPointer<Self> Pointer;

  itkNewMacro(Self);
  itkTypeMacro(CustomFilter, itk::ImageToImageFilter);

  /** Image typedefs */
  typedef typename TImageType::Pointer ImagePointer;
  typedef typename TImageType::PixelType PixelType;
  typedef typename TImageType::RegionType RegionType;
  typedef typename TImageType::SizeType SizeType;
  typedef typename TImageType::IndexType IndexType;

  /** Set/Get the radius of the neighborhood */
  itkSetMacro(Radius, unsigned int);
  itkGetMacro(Radius, unsigned int);

protected:
  CustomFilter() : m_Radius(1) {}
  ~CustomFilter() override = default;

  /** Main processing method */
  void GenerateData() override {
    // Get input and output images
    typename TImageType::ConstPointer inputImage = this->GetInput();
    ImagePointer outputImage = this->GetOutput();

    outputImage->SetRegions(inputImage->GetLargestPossibleRegion());
    outputImage->Allocate();
    outputImage->FillBuffer(0);

    // Define iterators
    itk::ImageRegionConstIterator<TImageType> inputIt(
        inputImage, inputImage->GetLargestPossibleRegion());
    itk::ImageRegionIterator<TImageType> outputIt(
        outputImage, outputImage->GetLargestPossibleRegion());

    SizeType radius;
    radius.Fill(m_Radius);

    // Process each pixel
    for (inputIt.GoToBegin(), outputIt.GoToBegin(); !inputIt.IsAtEnd();
         ++inputIt, ++outputIt) {
      IndexType centerIndex = inputIt.GetIndex();
      PixelType sum = 0;
      unsigned int count = 0;

      // Iterate over the neighborhood
      for (int dx = -static_cast<int>(m_Radius);
           dx <= static_cast<int>(m_Radius); ++dx) {
        for (int dy = -static_cast<int>(m_Radius);
             dy <= static_cast<int>(m_Radius); ++dy) {
          IndexType neighborIndex = centerIndex;
          neighborIndex[0] += dx;
          neighborIndex[1] += dy;

          if (inputImage->GetLargestPossibleRegion().IsInside(neighborIndex)) {
            sum += inputImage->GetPixel(neighborIndex);
            ++count;
          }
        }
      }

      // Compute the mean value
      outputIt.Set(static_cast<PixelType>(sum / count));
    }
  }

private:
  CustomFilter(const Self &) = delete;
  void operator=(const Self &) = delete;

  unsigned int m_Radius;
};
