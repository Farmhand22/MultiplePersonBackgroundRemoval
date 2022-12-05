// © Copyright 2022 Farmhand.

// Main Program file.
#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>
#pragma warning(disable: 4251 6294 6201 6269)
#include <libobsensor/hpp/Pipeline.hpp>
#include <libobsensor/hpp/Error.hpp>
#include "window.hpp"

import Const;
import HumanObjectTracker;
import FaceDetection;

// Constants
const std::string WINDOW_TITLE = "Multiple-Person Background Removal Using Orbbec Femto Developer Kit";
const cv::Scalar GREEN_SCREEN_COLOR(64, 177, 0);   // RGB: (0, 177, 64)

// Across threads.
static std::mutex gMutexFrames;
static std::shared_ptr<ob::FrameSet> gSpFrameSet;
static std::atomic<bool> gQuitApp{ false };

using namespace cv;
using namespace std::literals;


static std::vector<Mat> GetSynchronizedFrames(Window& app)
{
    std::vector<Mat> mats;

    // Limit the scope of the mutex lock.
    std::unique_lock<std::mutex> lock(gMutexFrames);
    auto colorFrame = gSpFrameSet->colorFrame();
    auto depthFrame = gSpFrameSet->depthFrame();
    if (colorFrame && depthFrame) {
        mats = app.processFrames({ colorFrame, depthFrame });
    }
    gSpFrameSet.reset();   // Reset for next set.
    return mats;
}

static void ProcessAndDisplayFrameSet(Window& app, HumanObjectTracker& hoTracker, FaceDetection& faceDet, TickMeter& tm)
{
    static int frameNumber = 0;

    if (!gSpFrameSet) {
        app.render({}, RenderType::RENDER_SINGLE);  // No image to display.
        return;
    }

    tm.start();
    frameNumber++;
    auto mats = GetSynchronizedFrames(app);
    if (2 != mats.size()) return;

    // 1. RGB image for face detection.
    Mat& imgColor = mats.at(0);
    const std::vector<Point2i>& faceCenters = faceDet.Detect(imgColor);

    // 2. Depth image for human object tracking.
    Mat& imgDepth = mats.at(1);    // Raw depth image. Mark it as the original depth image.
    const Mat& imgTrackerAsMask = hoTracker.ProcessFrameWithFaces(imgDepth, faceCenters);

    // 3. Copy original image to masked area to create output image.
    cv::Mat imgOut(imgColor.size(), CV_8UC3, GREEN_SCREEN_COLOR);
    imgColor.copyTo(imgOut, imgTrackerAsMask);

    // 3. Mark faces detected.
    tm.stop();
    faceDet.Visualize(imgColor, tm.getFPS(), 2);

    // RENDER_GRID needs all mats to be the same shape (480, 640, 3).
    cv::cvtColor(imgDepth, imgDepth, cv::COLOR_GRAY2RGB);
    putText(imgDepth, "Depth", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    putText(imgOut, "Output", Point(5, 15), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 2);
    app.renderMats({ imgColor, imgOut, imgDepth }, RenderType::RENDER_ONE_ROW);
}

int main() try
{
    //创建一个Pipeline，Pipeline是整个高级API的入口，通过Pipeline可以很容易的打开和关闭
    //多种类型的流并获取一组帧数据
    ob::Pipeline pipe;

    //获取彩色相机的所有流配置，包括流的分辨率，帧率，以及帧的格式
    auto colorProfiles = pipe.getStreamProfileList(OB_SENSOR_COLOR);

    //通过接口设置感兴趣项，返回对应Profile列表的首个Profile
    auto colorProfile = colorProfiles->getVideoStreamProfile(W, H, OB_FORMAT_RGB888, 30);
    if (!colorProfile) {
        colorProfile = colorProfiles->getProfile(0)->as<ob::VideoStreamProfile>();
    }

    //获取深度相机的所有流配置，包括流的分辨率，帧率，以及帧的格式
    auto depthProfiles = pipe.getStreamProfileList(OB_SENSOR_DEPTH);

    //通过接口设置感兴趣项，返回对应Profile列表的首个Profile
    auto depthProfile = depthProfiles->getVideoStreamProfile(W, H, OB_FORMAT_Y16, 30);
    if (!depthProfile) {
        depthProfile = depthProfiles->getProfile(0)->as<ob::VideoStreamProfile>();
    }

    //通过创建Config来配置Pipeline要启用或者禁用哪些流，这里将启用彩色流和深度流
    std::shared_ptr<ob::Config> config = std::make_shared<ob::Config>();
    config->enableStream(colorProfile);
    config->enableStream(depthProfile);

    // 配置对齐模式为软件D2C对齐
    config->setAlignMode(ALIGN_D2C_SW_MODE);

    //启动在Config中配置的流，如果不传参数，将启动默认配置启动流
    pipe.start(config);

    std::jthread waitFramesThread([&]() {
        while (!gQuitApp) {
            //以阻塞的方式等待一帧数据，该帧是一个复合帧，里面包含配置里启用的所有流的帧数据，
            //并设置帧的等待超时时间为100ms
            auto frameSet = pipe.waitForFrames(100);
            if (!frameSet) continue;
            std::unique_lock<std::mutex> lock(gMutexFrames, std::defer_lock);
            if (lock.try_lock()) {
                gSpFrameSet = frameSet;
            }
        }});

    TickMeter tm;
    FaceDetection faceDet(W, H);
    HumanObjectTracker hoTracker;
    //创建一个用于渲染的窗口，并设置窗口的分辨率
    Window app(WINDOW_TITLE, colorProfile->width() * 3, colorProfile->height());

    // Forever loop.
    while (app) {
        ProcessAndDisplayFrameSet(app, hoTracker, faceDet, tm);
        if (app.ScanKeyPress()) break;
    }

    gQuitApp = true;
    waitFramesThread.join();
    pipe.stop();
    return 0;
}
catch (const ob::Error& e)
{
    std::cerr << "Function:" << e.getName() << "\nArguments:" << e.getArgs() << "\nMessage:" << e.getMessage()
        << "\nType:" << e.getExceptionType() << std::endl;
    exit(EXIT_FAILURE);
}
