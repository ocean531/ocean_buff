#include "NNDetector.hpp"

#include <stdexcept>
#include <utility>

RuneDetector::RuneDetector(const JsonConfig &config,
                           std::size_t pipeline_delay,
                           const DebugConfig &debug)
    : m_model(config, debug), m_pipeline_delay(pipeline_delay)
{}


std::optional<RuneDetectionFrame> RuneDetector::process(const cv::Mat &image)
{
    if (image.empty())
        throw std::invalid_argument("能量机关检测不能输入空图像");  //空图检查

    //m_submitted_frames.push_back(image.clone()); //存入队列

    std::vector<NetRuneResult> results = m_model.netProcess(image); //推理

    //cv::Mat matched = takeMatchedFrame(m_submitted_frames, m_pipeline_delay); //异步匹配
    //if (matched.empty())
    //    return std::nullopt;
    //这里是同步推理，所以删除了异步匹配部分
    
    return RuneDetectionFrame{std::move(image), std::move(results)};
}