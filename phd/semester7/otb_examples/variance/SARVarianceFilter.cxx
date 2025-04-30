#include "itkVarianceImageFilter.h"
#include "otbImage.h"
#include "otbWrapperApplication.h"
#include "otbWrapperApplicationFactory.h"

class SARVarianceFilter : public otb::Wrapper::Application {
public:
  typedef SARVarianceFilter Self;
  typedef otb::Wrapper::Application Superclass;
  typedef itk::SmartPointer<Self> Pointer;
  itkNewMacro(Self);

private:
  void DoInit() override {
    // Application name and description
    SetName("SARVarianceFilter");
    SetDescription("Applies a variance filter on SAR images.");

    // Input image parameter
    AddParameter(ParameterType_InputImage, "in", "Input SAR Image");
    SetParameterDescription("in", "The SAR image to process.");

    // Output image parameter
    AddParameter(ParameterType_OutputImage, "out", "Output Image");
    SetParameterDescription(
        "out", "The resulting image after applying the variance filter.");

    // Radius parameter
    AddParameter(ParameterType_Int, "radius", "Filter Radius");
    SetParameterDescription("radius", "Radius of the variance filter kernel.");
    SetDefaultParameterInt("radius", 3);

    // Set roles
    SetParameterRole("in", Role_Input);
    SetParameterRole("out", Role_Output);
  }

  void DoExecute() override {
    // Define image type
    typedef otb::Image<float, 2> ImageType;

    // Read input image
    auto inputImage = GetParameterImage("in");

    // Define the variance filter
    typedef itk::VarianceImageFilter<ImageType, ImageType> VarianceFilterType;
    auto varianceFilter = VarianceFilterType::New();

    // Set the input image
    varianceFilter->SetInput(inputImage);

    // Set the radius
    VarianceFilterType::InputSizeType radius;
    radius.Fill(GetParameterInt("radius"));
    varianceFilter->SetRadius(radius);

    // Run the filter
    varianceFilter->Update();

    // Set the output image
    SetParameterOutputImage("out", varianceFilter->GetOutput());
  }
};

OTB_APPLICATION_EXPORT(SARVarianceFilter)
