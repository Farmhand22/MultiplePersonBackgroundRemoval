// © Copyright 2022 Farmhand.

module;
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>

export module FaceDetection;

using namespace cv;
using namespace std;

export class FaceDetection
{
private:
    const string FD_MODEL_PATH = "face_detection_yunet_2022mar.onnx";
    Ptr<FaceDetectorYN> mFaceDetector;
    Mat mFaces;  // Detection results in Mat, Rows == Faces.
    vector<Point2i> mFaceCenters;

public:
    explicit FaceDetection(const int frameWidth = 640, const int frameHeight = 480) try
        : mFaceDetector(FaceDetectorYN::create(FD_MODEL_PATH, "", Size(frameWidth, frameHeight))) {
    }
    catch (const std::exception& e) {
        cout << __FUNCTION__ << " ERROR: " << e.what() << endl;
    }

    // Return face centers.
    const vector<Point2i>& Detect(const Mat& imgColor);
    void Visualize(Mat& img, const double fps, const int thickness = 2) const;
};

module: private;

const vector<Point2i>& FaceDetection::Detect(const Mat& imgColor)
{
    mFaceDetector->detect(imgColor, mFaces);

    // Calculate the centers of all faces detected.
    mFaceCenters.clear();
    for (int i = 0; i < mFaces.rows; i++)
    {
        const auto x = mFaces.at<float>(i, 0) + mFaces.at<float>(i, 2) / 2;
        const auto y = mFaces.at<float>(i, 1) + mFaces.at<float>(i, 3) / 2;
        mFaceCenters.emplace_back(Point2i(static_cast<int>(x), static_cast<int>(y)));
    }
    return mFaceCenters;
}

void FaceDetection::Visualize(Mat& img, const double fps, const int thickness) const
{
    // Label image with frame rate.
    putText(img, cv::format("RGB  FPS: %.1f", fps), Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);

    const auto markColor = Scalar(0, 255, 0);
    for (int i = 0; i < mFaces.rows; i++) // For each face.
    {
        // Draw bounding box.
        rectangle(img, Rect2i(static_cast<int>(mFaces.at<float>(i, 0)), static_cast<int>(mFaces.at<float>(i, 1)),
            static_cast<int>(mFaces.at<float>(i, 2)), static_cast<int>(mFaces.at<float>(i, 3))), 
            markColor, thickness);
        // Draw center of face.
        circle(img, mFaceCenters.at(i), 2, markColor, thickness);
    }
}
