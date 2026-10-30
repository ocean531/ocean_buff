#include "interface.hpp"

#include "deployment_lib.hpp"

#include <stdexcept>
#include <utility>


JsonConfig::JsonConfig(std::string json_path,
                       std::string model_key,
                       std::string model_folder)
    : json_path(std::move(json_path)),
      model_key(std::move(model_key)),
      model_folder(std::move(model_folder))
{}


RuneModel::RuneModel(const JsonConfig &json_config,
                     const DebugConfig &debug_config)
    : m_model(std::make_unique<YOLOModel>(json_config, debug_config))
{
    if (!m_model->supportsRune())
        throw std::invalid_argument("模型后处理类型不是神符，无法构造 RuneModel");
}

RuneModel::~RuneModel() noexcept = default;

std::vector<NetRuneResult> RuneModel::netProcess(const cv::Mat &input_image)
{
    if (!m_model)
        return {};
    return m_model->netProcess(input_image);
}
