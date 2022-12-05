// © Copyright 2022 Farmhand.

module;  // global module fragment area. Put #include directives here 
#include <iostream>
#include <string>
#pragma warning(disable: 5054 6294 6201 6269)
#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>

// Interface
export module HumanObjectTracker;

import Const;
import Traverse4ConnectedNeighbors;

using namespace cv;

export class HumanObjectTracker
{
private:
    Mat mImgMaskAll;     // Mask for all persons. 0 and 255 binary image.

public:
    Mat& ProcessFrameWithFaces(const Mat& imgDepth, const std::vector<Point2i>& faceCenters);
};

module: private;

Mat& HumanObjectTracker::ProcessFrameWithFaces(const Mat& imgDepth, const std::vector<Point2i>& faceCenters)
{
    mImgMaskAll = Mat::zeros(imgDepth.size(), CV_8UC1);     // Start with a blank mask.

    for (const auto& faceCenter : faceCenters)
    {
        Mat imgMaskPerson = Mat::zeros(imgDepth.size(), CV_8UC1);  // Start with blank mask for each face.
        DetectConnectedComponent(imgDepth, faceCenter, imgMaskPerson);
        bitwise_or(mImgMaskAll, imgMaskPerson, mImgMaskAll);      // Combine into the overall mask.
    }
    
    return mImgMaskAll;
}
