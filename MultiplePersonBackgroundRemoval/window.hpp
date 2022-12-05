// Copyright(c) 2020 Orbbec Corporation. All Rights Reserved.
// Farmhand 11/2022: From Orbbec SDK. Optimized for speed and memory usage.

#pragma once

#include <string>
#pragma warning(push)
#pragma warning(disable: 5054)
#include <opencv2/opencv.hpp>
#include <libobsensor/ObSensor.hpp>
#pragma warning(pop)

constexpr int ESC = 27;

//快速取平方根的倒数
float Q_rsqrt(const float number) noexcept
{
    constexpr float threehalfs = 1.5F;
    const float x2 = number * 0.5F;
    float y = number;
    long i = *reinterpret_cast<long*>(&y);
    i = 0x5f3759df - (i >> 1);
    y = *reinterpret_cast<float*>(&i);
    y = y * (threehalfs - (x2 * y * y));

    return y;
}

enum class RenderType {
    RENDER_SINGLE,       //只渲染数组里的第一个帧
    RENDER_ONE_ROW,      //以一行的形式渲染数组里的帧
    RENDER_ONE_COLOUMN,  //以一列的形式渲染数组里的帧
    RENDER_GRID,         //以格子的形式渲染数组里的帧
    RENDER_OVERLAY       //以叠加的形式渲染数组里的帧
};

class Window {
private:
    std::string mTitle;
    int         mWidth;
    int         mHeight;
    bool        mWindowClose = false;
    int         mKeyPressed = -1;
    bool        mShowInfo = false;
    int         mAverageColorFps = 0;
    int         mAverageDepthFps = 0;
    int         mAverageIrFps = 0;

public:
    Window(const std::string& name, const int width, const int height) noexcept
        : mTitle(name),
        mWidth(width),
        mHeight(height) {
    }

    void resize(const int width, const int height) noexcept {
        mWidth = width;
        mHeight = height;
    }

    // Return true if ESC key pressed.
    bool ScanKeyPress()
    {
        mKeyPressed = cv::waitKey(1);
        if (mKeyPressed == ESC) {
            mWindowClose = true;
        }
        else if (mKeyPressed == 'I' || mKeyPressed == 'i') {
            mShowInfo = !mShowInfo;
        }

        if (mWindowClose) {
            cv::destroyAllWindows();
        }
        return mWindowClose;
    }

    void render(std::vector<std::shared_ptr<ob::Frame>> frames, const RenderType renderType) {
        if (ScanKeyPress()) return;

        auto mats = processFrames(frames);
        renderMats(mats, renderType);
    }

    void render(std::vector<std::shared_ptr<ob::Frame>> frames, const float alpha) {
        if (ScanKeyPress()) return;

        auto mats = processFrames(frames);
        renderMats(mats, alpha);
    }

    int getKey() const noexcept {
        return mKeyPressed;
    }

    operator bool() const noexcept {
        return !mWindowClose;
    }

    void setShowInfo(const bool show) noexcept {
        mShowInfo = show;
    };

    void setColorAverageFps(const int averageFps) noexcept {
        mAverageColorFps = averageFps;
    }

    void setDepthAverageFps(const int averageFps) noexcept {
        mAverageDepthFps = averageFps;
    }

    void setIrAverageFps(const int averageFps) noexcept {
        mAverageIrFps = averageFps;
    }

    std::vector<cv::Mat> processFrames(std::vector<std::shared_ptr<ob::Frame>> frames)
    {
        std::vector<cv::Mat> mats;
        if (mWindowClose) return mats;

        for (const auto& frame : frames) {
            if (!frame || frame->dataSize() < 1024) continue;

            auto videoFrame = frame->as<ob::VideoFrame>();

            int     averageFps = mAverageColorFps;
            cv::Mat rstMat;

            if (videoFrame->type() == OB_FRAME_COLOR) {
                if (videoFrame->format() == OB_FORMAT_MJPG) {
                    cv::Mat rawMat(1, videoFrame->dataSize(), CV_8UC1, videoFrame->data());
                    rstMat = cv::imdecode(rawMat, 1);
                }
                else if (videoFrame->format() == OB_FORMAT_NV21) {
                    cv::Mat rawMat(videoFrame->height() * 3 / 2, videoFrame->width(), CV_8UC1, videoFrame->data());
                    cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_NV21);
                }
                else if (videoFrame->format() == OB_FORMAT_YUYV || videoFrame->format() == OB_FORMAT_YUY2) {
                    cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC2, videoFrame->data());
                    cv::cvtColor(rawMat, rstMat, cv::COLOR_YUV2BGR_YUY2);
                }
                else if (videoFrame->format() == OB_FORMAT_RGB888) {
                    cv::Mat rawMat(videoFrame->height(), videoFrame->width(), CV_8UC3, videoFrame->data());
                    cv::cvtColor(rawMat, rstMat, cv::COLOR_RGB2BGR);
                }
            }
            else if (videoFrame->format() == OB_FORMAT_Y16 || videoFrame->format() == OB_FORMAT_YUYV
                || videoFrame->format() == OB_FORMAT_YUY2) {
                // IR or Depth Frame
                cv::Mat rawMat = cv::Mat(videoFrame->height(), videoFrame->width(), CV_16UC1, videoFrame->data());
                double scale;
                if (videoFrame->type() == OB_FRAME_DEPTH) {
                    scale = 1.0f / pow(2, videoFrame->pixelAvailableBitSize() - 10);
                }
                else {
                    scale = 1.0f / pow(2, videoFrame->pixelAvailableBitSize() - 8);
                }
                // 12 bit distance value, max 4095
                cv::convertScaleAbs(rawMat, rstMat, scale /* 0.0625==1/16==14bit */);  // Gray scale only.
                //cv::Mat cvtMat;
                //cv::convertScaleAbs(rawMat, cvtMat, scale /* 0.0625 */);
                //cv::cvtColor(cvtMat, rstMat, cv::COLOR_GRAY2RGB);
            }

            if (videoFrame->type() == OB_FRAME_DEPTH) {
                averageFps = mAverageDepthFps;
            }
            else if (videoFrame->type() == OB_FRAME_IR) {
                averageFps = mAverageIrFps;
            }

            if (mShowInfo) {
                drawInfo(rstMat, *videoFrame, averageFps);
            }
            mats.push_back(rstMat);
        }
        return mats;
    }

    static void drawInfo(cv::Mat& imageMat, ob::VideoFrame& frame, const int averageFps) {
        const auto fontColor = cv::Scalar(255, 0, 255);

        if (frame.type() == OB_FRAME_COLOR) {
            if (frame.format() == OB_FORMAT_NV21) {
                cv::putText(imageMat, "Color-NV21", cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
            }
            else if (frame.format() == OB_FORMAT_MJPG) {
                cv::putText(imageMat, "Color-MJPG", cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
            }
        }
        else if (frame.type() == OB_FRAME_DEPTH) {
            cv::putText(imageMat, "Depth", cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
        }
        else if (frame.type() == OB_FRAME_IR) {
            cv::putText(imageMat, "IR", cv::Point(8, 16), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1, 4, false);
        }

        if (frame.timeStamp()) {
            cv::putText(imageMat, ("Timestamp: " + std::to_string(frame.timeStamp())).c_str(),
                cv::Point(8, 40), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
        }
        cv::putText(imageMat, ("System timestamp: " + std::to_string(frame.systemTimeStamp())).c_str(),
            cv::Point(8, 64), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
        if (averageFps) {
            cv::putText(imageMat, ("Frame rate: " + std::to_string(averageFps)).c_str(),
                cv::Point(8, 88), cv::FONT_HERSHEY_SIMPLEX, 0.6, fontColor, 1);
        }
    }

    void renderMats(const std::vector<cv::Mat>& mats, const RenderType renderType) const try {
        if (mWindowClose || mats.empty()) return;

        cv::Mat outMat;
        switch (renderType)
        {
        case RenderType::RENDER_SINGLE:
            cv::imshow(mTitle, mats.at(0));
            break;

        case RenderType::RENDER_ONE_ROW:
            for (const auto& mat : mats) {
                if (outMat.dims > 0) {
                    cv::hconcat(outMat, mat, outMat);
                }
                else {
                    outMat = mat;
                }
            }
            cv::imshow(mTitle, outMat);
            break;

        case RenderType::RENDER_ONE_COLOUMN:
            for (const auto& mat : mats) {
                if (outMat.dims > 0) {
                    cv::vconcat(outMat, mat, outMat);
                }
                else {
                    outMat = mat;
                }
            }
            cv::imshow(mTitle, outMat);
            break;

        case RenderType::RENDER_GRID:
        {
            const auto    count = mats.size();
            const float   sq = 1.0f / Q_rsqrt(static_cast<float>(count));
            const int     isq = static_cast<int>(sq);
            const int     cols = (sq - isq < 0.01f) ? isq : isq + 1;
            const float   div = static_cast<float>(count) / cols;
            const int     idiv = static_cast<int>(div);
            const int     rows = (div - idiv < 0.01f) ? idiv : idiv + 1;
            for (int i = 0; i < rows; i++) {
                cv::Mat lineMat;
                for (int j = 0; j < cols; j++) {
                    cv::Mat mat;
                    const int index = i * cols + j;
                    if (index < count) {
                        mat = mats.at(index);
                    }
                    else {
                        const int numRows = mHeight / rows;
                        const int numCols = mWidth / cols;
                        mat = cv::Mat::zeros(numRows, numCols, CV_8UC3);
                    }
                    if (lineMat.dims > 0) {
                        cv::hconcat(lineMat, mat, lineMat);
                    }
                    else {
                        lineMat = mat;
                    }
                }
                if (outMat.dims > 0) {
                    cv::vconcat(outMat, lineMat, outMat);
                }
                else {
                    outMat = lineMat;
                }
            }
            cv::imshow(mTitle, outMat);
        }
        break;

        case RenderType::RENDER_OVERLAY:
            for (int i = 1; i < mats.size(); i++) {
                cv::bitwise_or(outMat, mats.at(i), outMat);
            }
            cv::imshow(mTitle, outMat);
            break;

        default:
            throw std::runtime_error(std::format("Render type not supported: {}\n", static_cast<int>(renderType)));
        }
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }

    void renderMats(const std::vector<cv::Mat>& mats, const float alpha) const try {
        if (mWindowClose) return;

        const auto size = mats.size();
        if (size != 2) {
            std::cout << __FUNCTION__ ": Only support the saturation with two mats!" << std::endl;
            return;
        }

        const float invAlpha = 1.0f - alpha;
        cv::Mat outMat, resizeMat;
        cv::resize(mats.at(0), outMat, cv::Size(mWidth, mHeight));
        cv::resize(mats.at(1), resizeMat, cv::Size(mWidth, mHeight));
        for (int i = 0; i < outMat.rows; i++) {
            for (int j = 0; j < outMat.cols; j++) {
                cv::Vec3b& outRgb = outMat.at<cv::Vec3b>(i, j);
                const cv::Vec3b& resizeRgb = resizeMat.at<cv::Vec3b>(i, j);
                outRgb[0] = static_cast<uint8_t>(outRgb[0] * invAlpha + resizeRgb[0] * alpha);
                outRgb[1] = static_cast<uint8_t>(outRgb[1] * invAlpha + resizeRgb[1] * alpha);
                outRgb[2] = static_cast<uint8_t>(outRgb[2] * invAlpha + resizeRgb[2] * alpha);
            }
        }
        cv::imshow(mTitle, outMat);
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
};
