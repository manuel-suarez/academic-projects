//
// Created by masuarez on 20/10/22.
//
#include <opencv2/opencv.hpp>
#include <cmath>
#include <iomanip>
#include <iostream>

using namespace cv;
using namespace std;

using namespace std;

void PrintMatValues(const Mat &mat, int size, String msg = "Check values") {
    cout << msg << endl;
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            Point pt(i, j);
            cout << pt << ", " << mat.at<float>(pt) << "\t";
        }
        cout << endl;
    }
    cout << endl;
}

Mat FilterCreation(int size)
{
    // initialising standard deviation to 1.0
    double sigma = 0.3*((size-1)*0.5 - 1) + 0.8;
    double r, s = 2 * sigma * sigma;

    // sum is for normalization
    double sum = 0.0;

    // result
    Mat result = Mat::zeros(size, size, CV_32F);
    //PrintMatValues(result, size, "Check values 0");

    // generating size x size kernel
    for (int x = -size/2; x <= size/2; x++) {
        for (int y = -size/2; y <= size/2; y++) {
            r = sqrt(x * x + y * y);
            Point pt(x + size/2, y + size/2);
            result.at<float>(pt) = (exp(-(r * r) / s)) / (M_PI * s);
            sum += result.at<float>(pt);
        }
    }

    //PrintMatValues(result, size, "Check values 1");

    // normalization
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            Point pt = Point(i, j);
            result.at<float>(pt) = result.at<float>(pt) / sum;
        }
    }

    //PrintMatValues(result, size, "Check values 2");
    return result.clone();
}

void ApplyGaussianBlur(const Mat& mat, Mat& dest, const Mat& kernel, int size) {
    //PrintMatValues(kernel, size, "Check values kernel");
    cout << "Apply gaussian blur size: " << size << endl;
    dest = Mat::zeros(mat.rows, mat.cols, mat.type());

    for (int x = 0; x < mat.rows; x++) {
        for (int y = 0; y < mat.cols; y++) {
            // Aplicamos kernel en el punto p con su vencidad N(q)
            if (x < size/2 || y < size/2 || (x >= mat.rows - size/2) || (y >= mat.cols - size/2)) {
                Vec3b intensity = mat.at<Vec3b>(x, y);
                dest.at<Vec3b>(x, y) = intensity;
                continue;
            }
            // Calculamos kernel
            double tc1, tc2, tc3;
            tc1 = tc2 = tc3 = 0.0;
            for (int i = -size/2; i <= size/2; i++) {
                for (int j = -size/2; j <= size/2; j++) {
                    // Obtenemos intensidad de color en la posición (x+i,y+j)
                    Vec3b q_intensity = mat.at<Vec3b>(x+i, y+j);
                    tc1 += q_intensity.val[0]/255.0 * kernel.at<float>(i+size/2, j+size/2);
                    tc2 += q_intensity.val[1]/255.0 * kernel.at<float>(i+size/2, j+size/2);
                    tc3 += q_intensity.val[2]/255.0 * kernel.at<float>(i+size/2, j+size/2);
                }
            }
            Vec3b intensity(tc1 * 255.0, tc2 * 255.0, tc3 * 255.0);
            dest.at<Vec3b>(x, y) = intensity;
        }
    }
}

void ApplyGaussianBlurAndShow(const Mat& image, int size)
{
    // Blur the image with 3x3 Gaussian kernel
    Mat cv_image_blurred;
    Mat custom_image_blurred;
    Mat kernel = FilterCreation(size);
    //PrintMatValues(kernel_3x3, 5, "Check values main");
    ApplyGaussianBlur(image, custom_image_blurred, kernel, size);
    GaussianBlur(image, cv_image_blurred, Size(size, size), 0);
    Mat kernel_cv = getGaussianKernel(size, 0);
    //PrintMatValues(kernel_cv, size, "Check cv kernel");

    // Window's names
    stringstream window_name_blurred_cv;
    stringstream window_name_blurred_custom;
    window_name_blurred_cv << "Lotus Blurred with " << size << " x " << size << " Gaussian Kernel";
    window_name_blurred_custom << "Lotus Blurred with " << size << " x " << size << " Gaussian Kernel (custom)";

    // Create windows with above names
    namedWindow(window_name_blurred_cv.str());
    namedWindow(window_name_blurred_custom.str());

    // Show our images inside the created windows.
    imshow(window_name_blurred_cv.str(), cv_image_blurred);
    imshow(window_name_blurred_custom.str(), custom_image_blurred);
}

int main()
{
    // Read image file
    Mat image = imread("../Blue-Lotus.jpg");

    // Check for failure
    if (image.empty())
    {
        cout << "Could not open or find the image" << endl;
        cin.get(); // wait for any key press
        return -1;
    }

    cout << "Image channels: " << image.channels() << endl;
    cout << "Image dimensions: " << image.dims << endl;
    cout << "Image type: " << image.type() << endl;
    cout << "Image size: " << image.size << endl;
    double min, max;
    cv::minMaxLoc(image, &min, &max);
    cout << "Min value: " << min << endl;
    cout << "Max value: " << max << endl;

    // Windows original image
    String window_name = "Lotus";

    // Create window
    namedWindow(window_name);

    // Show original image
    imshow(window_name, image);

    // Blur the image with 3x3, 5x5 Gaussian kernel
    ApplyGaussianBlurAndShow(image, 3);
    ApplyGaussianBlurAndShow(image, 5);
    ApplyGaussianBlurAndShow(image, 9);

    waitKey(0); // Wait stroke
    destroyAllWindows();

    return 0;
}