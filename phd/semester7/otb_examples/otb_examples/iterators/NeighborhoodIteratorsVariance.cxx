// This example uses the \doxygen{itk}{NeighborhoodIterator} to implement a
// simple Sobel edge detection algorithm \cite{Gonzalez1993}.  The algorithm
// uses the neighborhood iterator to iterate through an input image and
// calculate a series of finite difference derivatives.  Since the derivative
// results cannot be written back to the input image without affecting later
// calculations, they are written instead to a second, output image.  Most
// neighborhood processing algorithms follow this read-only model on their
// inputs.
//
// We begin by including the proper header files.  The
// \doxygen{itk}{ImageRegionIterator} will be used to write the results of
// computations to the output image.  A const version of the neighborhood
// iterator is used because the input image is read-only.

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

  // The following code creates and executes the OTB image reader.
  // The \code{Update}
  // call on the reader object is surrounded by the standard \code{try/catch}
  // blocks to handle any exceptions that may be thrown by the reader.

  ReaderType::Pointer reader = ReaderType::New();
  reader->SetFileName(argv[1]);
  try {
    reader->Update();
  } catch (itk::ExceptionObject &err) {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
  }

  //  We can now create a neighborhood iterator to range over the output of the
  //  reader. For Sobel edge-detection in 2D, we need a square iterator that
  //  extends one pixel away from the neighborhood center in every dimension.

  NeighborhoodIteratorType::RadiusType radius;
  radius.Fill(3);
  NeighborhoodIteratorType it(radius, reader->GetOutput(),
                              reader->GetOutput()->GetRequestedRegion());

  // The following code creates an output image and iterator.

  ImageType::Pointer output = ImageType::New();
  output->SetRegions(reader->GetOutput()->GetRequestedRegion());
  output->Allocate();

  IteratorType out(output, reader->GetOutput()->GetRequestedRegion());

  for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out) {
    float mean = 0.0;
    float sum = 0.0;
    unsigned int count = 0;

    // Iterate through the neighborhood to calculate the mean
    for (unsigned int i = 0; i < it.Size(); ++i) {
      if (it.GetPixel(i) != itk::NumericTraits<PixelType>::Zero) {
        sum += it.GetPixel(i);
        ++count;
      }
    }
    mean = (count > 0) ? sum / count : 0.0;
    // Iterate through the neighborhood to calculate the standard deviation
    sum = 0;
    count = 0;
    for (unsigned int i = 0; i < it.Size(); ++i) {
      if (it.GetPixel(i) != itk::NumericTraits<PixelType>::Zero) {
        float aux = it.GetPixel(i) - mean;
        sum += aux * aux;
        ++count;
      }
    }
    float std = (count > 0) ? sum / count : 0.0;
    out.Set(std);
  }

  // The last step is to write the output buffer to an image file.  Writing is
  // done inside a \code{try/catch} block to handle any exceptions.  The output
  // is rescaled to intensity range $[0, 255]$ and cast to unsigned char so that
  // it can be saved and visualized as a PNG image.

  using WritePixelType = float;
  using WriteImageType = otb::Image<WritePixelType, 2>;
  using WriterType = otb::ImageFileWriter<WriteImageType>;

  WriterType::Pointer writer = WriterType::New();
  writer->SetFileName(argv[2]);
  writer->SetInput(output);
  try {
    writer->Update();
  } catch (itk::ExceptionObject &err) {
    std::cout << "ExceptionObject caught !" << std::endl;
    std::cout << err << std::endl;
    return -1;
  }

  // The center image of Figure~\ref{fig:NeighborhoodExamples1} shows the
  // output of the Sobel algorithm applied to
  // \code{Examples/Data/ROI\_QB\_PAN\_1.tif}.
  //
  // \begin{figure} \centering
  // \includegraphics[width=0.45\textwidth]{ROI_QB_PAN_1.eps}
  // \includegraphics[width=0.45\textwidth]{NeighborhoodIterators1a.eps}
  // \itkcaption[Sobel edge detection results]{Applying the Sobel operator to an
  // image (left) produces $x$ (right)  derivative image.}
  // \protect\label{fig:NeighborhoodExamples1}
  // \end{figure}

  return EXIT_SUCCESS;
}
