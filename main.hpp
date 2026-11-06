#pragma once

#include "NNDetector.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>

#include "buff_algo/type.hpp"


namespace fs = std::filesystem;

namespace
{
struct VideoResult
{
    std::uint64_t frames = 0;
    fs::path output_path;
};

struct VideoClip
{
    cv::VideoCapture video;
    std::uint64_t frame_limit = 0;
};

// 展示片段时长（秒）。
constexpr double kShowcaseSeconds = 10.0;

fs::path projectRoot(int argc, char **argv)
{
    return fs::absolute(argc > 1 ? fs::path(argv[1])
                                 : fs::path("."));
}

cv::VideoCapture openVideo(const fs::path &path)
{
    cv::VideoCapture video(path.string());
    if (!video.isOpened())
        throw std::runtime_error("无法打开测试视频: " + path.string());
    return video;
}

VideoClip openMiddleClip(const fs::path &path)
{
    cv::VideoCapture video = openVideo(path);
    const double fps = video.get(cv::CAP_PROP_FPS);
    const double frame_count = video.get(cv::CAP_PROP_FRAME_COUNT);
    if (!std::isfinite(fps) || fps <= 0.0 ||
        !std::isfinite(frame_count) || frame_count <= 0.0)
        throw std::runtime_error("测试视频帧率或帧数无效: " + path.string());

    const double clip_frames = std::min(frame_count, kShowcaseSeconds * fps);
    const double start_frame = std::max(0.0, (frame_count - clip_frames) / 2.0);
    if (!video.set(cv::CAP_PROP_POS_FRAMES, std::floor(start_frame)))
        throw std::runtime_error("无法定位测试视频中段: " + path.string());
    return {std::move(video), static_cast<std::uint64_t>(std::floor(clip_frames))};
}

cv::VideoWriter createWriter(const cv::VideoCapture &video, const fs::path &output_path)
{
    const double fps = video.get(cv::CAP_PROP_FPS);
    const int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::VideoWriter writer(output_path.string(),
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(width, height));
    if (!writer.isOpened())
        throw std::runtime_error("无法创建输出视频: " + output_path.string());
    return writer;
}

cv::Point point(const cv::Point2d &value)
{
    return {cvRound(value.x), cvRound(value.y)};
}
void drawLabel(cv::Mat &image, const std::string &text, cv::Point origin,
               double scale = 0.45)
{
    int baseline = 0;
    const cv::Size size = cv::getTextSize(
        text, cv::FONT_HERSHEY_SIMPLEX, scale, 1, &baseline);
    origin.x = std::clamp(origin.x, 0, std::max(0, image.cols - size.width - 6));
    origin.y = std::clamp(origin.y, size.height + 6,
                          std::max(size.height + 6, image.rows - baseline - 2));
    const cv::Rect background(
        origin.x, origin.y - size.height - 5,
        size.width + 6, size.height + baseline + 7);
    cv::rectangle(image, background, cv::Scalar(24, 24, 24), cv::FILLED);
    cv::rectangle(image, background, cv::Scalar(255, 255, 255), 1);
    cv::putText(image, text, origin + cv::Point(3, -2),
                cv::FONT_HERSHEY_SIMPLEX, scale, cv::Scalar(255, 255, 255),
                1, cv::LINE_AA);
}

void drawIndexedPoint(cv::Mat &image, const cv::Point2d &value,
                      std::size_t index, const cv::Scalar &color)
{
    const cv::Point center = point(value);
    cv::circle(image, center, 4, color, cv::FILLED, cv::LINE_AA);
    drawLabel(image, std::to_string(index), center + cv::Point(7, -5), 0.4);
}

//绘制关键点
void drawRune(cv::Mat &image, const std::vector<NetRuneResult> &results)
{
    constexpr std::array<int, 4> outline = {0, 1, 4, 3};
    for (const NetRuneResult &result : results)
    {
        if (result.points.size() != 5)
            continue;
        for (std::size_t index = 0; index < outline.size(); ++index)
            cv::line(image, point(result.points[outline[index]]),
                     point(result.points[outline[(index + 1) % outline.size()]]),
                     cv::Scalar(255, 255, 0), 2, cv::LINE_AA);
        drawLabel(image, "class=" + std::to_string(result.class_id),
                  point(result.points.front()) + cv::Point(0, -28), 0.5);
        for (std::size_t index = 0; index < result.points.size(); ++index)
            drawIndexedPoint(image, result.points[index], index,
                             cv::Scalar(0, 255, 255));
    }
}


//推理能量机关
std::optional<RuneDetectionFrame> runRune
                   (const fs::path &root = fs::path{}, const fs::path &config_path,
                    const fs::path &output_dir = fs::path{},
                    std::size_t pipeline_delay,
                    cv::Mat input_ = {})
{
    const fs::path models = root / "所有模型/openvino";
    RuneDetector detector(
        JsonConfig{config_path.string(), "rune_detect", models.string()},
        pipeline_delay);

    std::optional<RuneDetectionFrame> last_frame;
    std::optional<RuneDetectionFrame> output;

    if(input_.empty())
    { //使用视频测试时
    VideoClip clip = openMiddleClip(root / "测试视频/符.avi");
    cv::VideoWriter writer = createWriter(clip.video, output_path);
    const fs::path output_path = output_dir / "rune_result.mp4";

    std::uint64_t count = 0;
    std::uint64_t submitted = 0;
    cv::Mat input;
    while (submitted < clip.frame_limit && clip.video.read(input))
    {
        ++submitted;
        output = detector.process(input);
        if (!output)
            continue;
        last_frame = std::move(output);
        drawRune(last_frame->image, last_frame->results);
        cv::imshow("img", last_frame->image);
        if (cv::waitKey(1) == 27)  // ESC 键退出
            break;
        writer.write(last_frame->image);
        ++count;
    }
    cv::destroyAllWindows();
    return last_frame;
    };

    else
    {//真实使用时
        while(1){
        cv::VideoWriter writer = createWriter(clip.video, output_path);
        output = detector.process(input_);
        if (!output)
            continue;
        last_frame = std::move(output);
        drawRune(last_frame->image, last_frame->results);
        cv::imshow("img", last_frame->image);
        if (cv::waitKey(1) == 27)  // ESC 键退出
            break;
        
    
        cv::destroyAllWindows();
        return last_frame;
        };
    };

}


}