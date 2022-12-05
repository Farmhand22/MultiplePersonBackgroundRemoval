// © Copyright 2022 Farmhand.

module;  // global module fragment area. Put #include directives here 
#include <iostream>
#include <vector>
#include <thread>
#include <queue>
#pragma warning(disable: 5054 6294 6201 6269)
#include <opencv2/opencv.hpp>

// Interface
export module Traverse4ConnectedNeighbors;

import Const;

//#define SHOW_4_CONNECTED_TRAVERSE   // Uncomment to show the visualization of traversing 4-connected neighbors.

// Assume Moving speed at any direction max: 100 cm/s
constexpr int CONNECTED_THRESHOLD = 12;    // => n*16mm. Based on human body contour. 
constexpr int MARK_BINARY = 255;   // Mark for the connected zones on a binary image.

using namespace cv;

// Return list of 4-connected neighbor points. 
static std::vector<Point2i>& List4ConnectedNeighbors(const Point2i& centerPoint)
{
    static std::vector<Point2i> neighbors(4);  // Static variable for speed.

    neighbors.clear();
    // Check edge point.
    if (centerPoint.x > 0)
        neighbors.emplace_back(Point2i(centerPoint.x - 1, centerPoint.y));
    if (centerPoint.y > 0)
        neighbors.emplace_back(Point2i(centerPoint.x, centerPoint.y - 1));
    if (centerPoint.x < W_1)
        neighbors.emplace_back(Point2i(centerPoint.x + 1, centerPoint.y));
    if (centerPoint.y < H_1)
        neighbors.emplace_back(Point2i(centerPoint.x, centerPoint.y + 1));
    return neighbors;
}

#ifdef SHOW_4_CONNECTED_TRAVERSE
// Accumulate checked zones.
static void DisplayZonesChecked(const Point2i& zone, Mat& imgZonesConnected)
{
    static bool drawing = true;
    if (!drawing) return;

    using namespace std::chrono_literals;
    imgZonesConnected.at<uint8_t>(zone.y, zone.x) = MARK_BINARY;

    constexpr int zonesPerStep = 64;
    static int counter = 0;
    counter++;
    if ((counter < zonesPerStep) || (0 == (counter % zonesPerStep)))
    {
        cv::imshow("4-Connected Zones", imgZonesConnected);
        if (27 == cv::waitKey(1)) drawing = false;
    }
}
#endif

// Traverse 4-connected neighbors from a center point.
export void DetectConnectedComponent(const Mat& imgDepth, const Point2i& center, Mat& imgConnectedMask)
{
    if (imgDepth.at<uint8_t>(center.y, center.x) == 0) return;  // Nothing there to detect.

    static std::queue<Point2i> listToCheck;
    assert(listToCheck.empty() && "List of points to check must be empty to start with.");
    imgConnectedMask.at<uint8_t>(center.y, center.x) = MARK_BINARY;   // Initial center marked.
    listToCheck.push(center);  // Starting point

#ifdef SHOW_4_CONNECTED_TRAVERSE
    Mat imgZonesConnected = Mat::zeros(imgDepth.size(), CV_8UC1);   // A new image for each frame.
#endif

    // Loop through all connected zones. Time consuming.
    while (listToCheck.size() > 0)
    {
        const Point2i& centerPoint = listToCheck.front();  // Ref for speed. Get one zone at the front of the queue.
#ifdef SHOW_4_CONNECTED_TRAVERSE
        DisplayZonesChecked(centerPoint, imgZonesConnected);
#endif
        const int centerDistance = imgDepth.at<uint8_t>(centerPoint.y, centerPoint.x);
        for (const auto& pt : List4ConnectedNeighbors(centerPoint))
        {
            uint8_t& zoneByte = imgConnectedMask.at<uint8_t>(pt.y, pt.x);
            if (zoneByte) continue; // Already checked.
            if (abs(centerDistance - imgDepth.at<uint8_t>(pt.y, pt.x)) <= CONNECTED_THRESHOLD)
            {
                listToCheck.push(pt);
                zoneByte = MARK_BINARY;   // marked as connected
            }
        }
        listToCheck.pop();
    }
}
