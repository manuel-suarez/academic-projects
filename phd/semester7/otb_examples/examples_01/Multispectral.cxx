#include "itkShiftScaleImageFilter.h"
#include "otbImageFileReader.h"
#include "otbImageFileWriter.h"
#include "otbMultiToMonoChannelExtractROI.h"
#include "otbPerBandVectorImageFilter.h"
#include <cstdlib>

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cerr << "Usage: " << argv[0]
              << " <input_filename> <output_extract> <output_shifted_scaled>"
              << std::endl;
  }

  using PixelType = unsigned short;
  using VectorImageType = otb::VectorImage<PixelType, 2>;

  using ReaderType = otb::ImageFileReader<VectorImageType>;
  ReaderType::Pointer reader = ReaderType::New();

  reader->SetFileName(argv[1]);

  using ExtractChannelType =
      otb::MultiToMonoChannelExtractROI<PixelType, PixelType>;
  ExtractChannelType::Pointer extractChannel = ExtractChannelType::New();

  reader->UpdateOutputInformation();
  extractChannel->SetExtractionRegion(
      reader->GetOutput()->GetLargestPossibleRegion());
  extractChannel->SetChannel(3);
  extractChannel->SetInput(reader->GetOutput());

  using ImageType = otb::Image<PixelType, 2>;
  using WriterType = otb::ImageFileWriter<ImageType>;
  WriterType::Pointer writer = WriterType::New();

  writer->SetFileName(argv[2]);
  writer->SetInput(extractChannel->GetOutput());

  writer->Update();

  using ShiftScaleType = itk::ShiftScaleImageFilter<ImageType, ImageType>;
  ShiftScaleType::Pointer shiftScale = ShiftScaleType::New();
  shiftScale->SetScale(0.5);
  shiftScale->SetShift(10);

  using VectorFilterType =
      otb::PerBandVectorImageFilter<VectorImageType, VectorImageType,
                                    ShiftScaleType>;
  VectorFilterType::Pointer vectorFilter = VectorFilterType::New();
  vectorFilter->SetFilter(shiftScale);
  vectorFilter->SetInput(reader->GetOutput());

  using VectorWriterType = otb::ImageFileWriter<VectorImageType>;
  VectorWriterType::Pointer writerVector = VectorWriterType::New();

  writerVector->SetFileName(argv[3]);
  writerVector->SetInput(vectorFilter->GetOutput());

  writerVector->Update();

  return EXIT_SUCCESS;
}
