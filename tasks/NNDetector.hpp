#pragma once

#include "interface.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <vector>



struct RuneDetectionFrame
{
    cv::Mat image;
    std::vector<NetRuneResult> results;
};

class RuneDetector
{
public:
    explicit RuneDetector(const JsonConfig &config,
                          std::size_t pipeline_delay,
                          const DebugConfig &debug = DebugConfig());

    // pipeline_delay 由调用方根据推理流水线设置；预热期间返回 std::nullopt。
    // 预热后，image 与 results 始终属于同一输入帧。
    std::optional<RuneDetectionFrame> process(const cv::Mat &image);

private:
    RuneModel m_model;  //模型
    std::size_t m_pipeline_delay; //延迟
    //std::deque<cv::Mat> m_submitted_frames; //帧缓冲队列
};
