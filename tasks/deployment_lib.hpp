#pragma once

#include "interface.hpp"
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace MPT { struct SpeedStats; }


struct InferParam
{
    int origin_width = 0;
    int origin_height = 0;
    int out_tensor_cols = 0;
    int out_tensor_rows = 0;
    float scale = 1.0;
    int pad_x = 0;
    int pad_y = 0;
};



class YOLOModel
{
public:

    // 直接指定模型路径、推理模式和部署后端。
    // YOLOModel(const std::string &model_path,
    //           const std::string &infer_mode,
    //           const std::string &deploy_way,
    //           const std::string &device = "GPU",
    //           float confidence_threshold = 0.5f,
    //           const std::string &postprocess_mode = "auto_detect",
    //           const DebugConfig &debug_config = DebugConfig());

    // 从 JSON 的指定节点读取模型配置。
    YOLOModel(const JsonConfig &json_config,
              const DebugConfig &debug_config = DebugConfig());

    //YOLOModel() = delete;
    
    ~YOLOModel() noexcept;

    YOLOModel(const YOLOModel &) = delete;
    YOLOModel &operator=(const YOLOModel &) = delete;
    YOLOModel(YOLOModel &&) noexcept = default;
    YOLOModel &operator=(YOLOModel &&) noexcept = default;

    std::vector<NetRuneResult> netProcess(const cv::Mat &input_image);

    // 判断当前模型是否使用神符后处理。
    bool supportsRune() const;
    class InferenceEngine;
    class PostProcessor;

    enum class NetDeployWay
    {
        openvino = 0,
        tensorrt = 1
    };
    enum class NetInferMode
    {
        sync = 0,
        async = 1,
        async4 = 2
    };
    enum class NetPostProcessMode
    {
        auto_detect = 0,
        v5infantry_fourpoints = 1,
        v8infantry_fourpoints = 2,
        lidar_fourpoints = 3,
        rune_fivepoints = 4,
        v8infantry_fourpoints_21 = 5
    };

    struct ModelConfig
    {
        std::string model_path;
        NetPostProcessMode postprocess_mode;
        NetInferMode infer_mode;
        NetDeployWay deploy_way;
        std::string device;
        float confidence_threshold = 0.5f;

        ModelConfig() = delete;
        ModelConfig(const std::string &path,
                    const NetInferMode &infer_mode = NetInferMode::async,
                    const NetDeployWay &deploy_way = NetDeployWay::openvino,
                    const std::string &device = "GPU",
                    float confidence = 0.5f,
                    const NetPostProcessMode &postprocess_mode = NetPostProcessMode::auto_detect)
            : model_path(path),
              postprocess_mode(postprocess_mode),
              infer_mode(infer_mode),
              deploy_way(deploy_way),
              device(device),
              confidence_threshold(confidence)
        {}
    };

private:
    YOLOModel(const ModelConfig &model_config, const DebugConfig &debug_config = DebugConfig());

    static float defaultNmsThreshold(NetPostProcessMode mode);

    ModelConfig m_model_config;
    DebugConfig m_debug_config;
    std::unique_ptr<MPT::SpeedStats> m_speed_stats;

    std::unique_ptr<InferenceEngine> m_inference_engine;
    std::unique_ptr<PostProcessor> m_postprocessor;
};

