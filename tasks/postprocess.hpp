#pragma once

#include "deployment_lib.hpp"

#include <opencv2/core.hpp>

#include <queue>
#include <string>
#include <vector>

class YOLOModel::PostProcessor
{
public:
    PostProcessor(const ModelConfig &model_config, float nms_threshold);
    virtual ~PostProcessor() = default;

    virtual std::vector<NetRuneResult> postProcessRuneMat(const float *input_ptr, const InferParam &infer_param) { throw std::logic_error("detect postprocess not supported"); };

protected:
    ModelConfig m_model_config;
    float m_nms_threshold = 0.f;
};



class RunePostProcessor : public YOLOModel::PostProcessor
{
public:
    RunePostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold);
    ~RunePostProcessor() = default;

    std::vector<NetRuneResult> postProcessRuneMat(const float *input_ptr, const InferParam &infer_param) override;

private:
    std::vector<std::string> m_class_names = {"未激活", "小符已激活", "大符已激活"};
};
