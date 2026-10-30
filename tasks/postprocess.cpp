#include "postprocess.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <numeric>
#include <queue>
#include <sstream>
#include <utility>

// ==================== 后处理模块 ====================
// 后处理模块：v5 armor、v8 armor、lidar 和 rune 四类结果解析。

// ==================== 部署流程辅助区域 ====================

// 将每个候选项封装，便于直接进行张量计算
struct CandidateView
{
    const float *data = nullptr;
    std::size_t candidate_index = 0;
    std::size_t candidate_stride = 0;
    std::size_t feature_stride = 0;

    // 按特征索引读取当前候选项的数据。
    float operator[](std::size_t feature_index) const noexcept
    {
        return data[candidate_index * candidate_stride + feature_index * feature_stride];
    }
};



// 保存后处理所需的模型配置和非极大值抑制阈值。
YOLOModel::PostProcessor::PostProcessor(const ModelConfig &model_config, float nms_threshold)
    : m_model_config(model_config), m_nms_threshold(nms_threshold) {}

// ==================== 符实现 ====================
// 初始化神符后处理器。
RunePostProcessor::RunePostProcessor(const YOLOModel::ModelConfig &model_config, float nms_threshold)
    : PostProcessor(model_config, nms_threshold) {}

namespace
{
// 按符叶四个边缘点的中心距离执行非极大值抑制。
void centerDistanceNMS(
    const std::vector<cv::Point2d> &keypoints_list, // 每个关键点: (x, y)
    const std::vector<float> &confidences_list,
    const std::vector<int> &class_ids_list,
    float center_dist_threshold,
    std::vector<int> &indices)
{
    indices.clear();

    // 检查输入
    if (keypoints_list.empty() || confidences_list.empty() || class_ids_list.empty())
    {
        return;
    }

    // 每个检测应该有5个关键点
    const int POINTS_PER_DETECTION = 5;
    int num_detections = confidences_list.size();

    if (keypoints_list.size() != static_cast<std::size_t>(num_detections * POINTS_PER_DETECTION))
    {
        std::cerr << "NMS时关键点数目错误" << std::endl;
        return;
    }

    // 1. 计算每个检测的中心点（使用符叶4个角点的平均值）
    std::vector<cv::Point2f> centers;
    centers.reserve(num_detections);

    for (int i = 0; i < num_detections; ++i)
    {
        // 获取当前检测的5个关键点
        const cv::Point2d *kpts = &keypoints_list[i * POINTS_PER_DETECTION];

        // 跳过索引2的R点，使用 top、left、right、bottom 四点计算中心。
        cv::Point2f center(0, 0);
        for (int k = 0; k < 5; ++k)
        {
            // 跳过R标
            if (k == 2)
                continue;
            center.x += kpts[k].x;
            center.y += kpts[k].y;
        }
        center.x /= 4.0f;
        center.y /= 4.0f;
        centers.push_back(center);
    }
    // 2. 初始化抑制标记
    std::vector<bool> suppressed(num_detections, false);

    // 3. 距离阈值平方（避免开方计算）
    float dist_threshold_sq = center_dist_threshold * center_dist_threshold;

    // 4. 按类别分组，但保持原indices不变，只给每个小符和大符打标签
    std::vector<int> sorted_indices(num_detections);
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&confidences_list](int a, int b)
              {
                  return confidences_list[a] > confidences_list[b];
              });

    auto same_group = [&class_ids_list](int a, int b)
    {
        bool a_lt_2 = class_ids_list[a] < 2;
        bool b_lt_2 = class_ids_list[b] < 2;
        bool a_eq_2 = class_ids_list[a] == 2;
        bool b_eq_2 = class_ids_list[b] == 2;
        return (a_lt_2 && b_lt_2) || (a_eq_2 && b_eq_2);
    };

    // 5. 同组间按置信度降序排序（保持索引），不同组不比较
    for (size_t i = 0; i < sorted_indices.size(); ++i)
    {
        int idx_i = sorted_indices[i];

        if (suppressed[idx_i])
            continue;

        indices.push_back(idx_i);
        const cv::Point2f &center_i = centers[idx_i];

        for (size_t j = i + 1; j < sorted_indices.size(); ++j)
        {
            int idx_j = sorted_indices[j];

            if (suppressed[idx_j])
                continue;

            // 不在同一组，不能互相抑制
            if (!same_group(idx_i, idx_j))
                continue;

            const cv::Point2f &center_j = centers[idx_j];
            float dx = center_i.x - center_j.x;
            float dy = center_i.y - center_j.y;
            float dist_sq = dx * dx + dy * dy;

            // 大符的nms距离阈值是正常小符/未激活的1/3，这里平方了所以除9
            float current_dist_threshold_sq = dist_threshold_sq;
            if (class_ids_list[idx_i] == 2)
            {
                current_dist_threshold_sq = dist_threshold_sq / 9.0f;
            }

            if (dist_sq < current_dist_threshold_sq)
            {
                suppressed[idx_j] = true;
            }
        }
    }
}
} // namespace

// 解析神符模型输出并生成符叶检测结果。
std::vector<NetRuneResult> RunePostProcessor::postProcessRuneMat(const float *input_ptr, const InferParam &infer_param)
{
    const float confidence_threshold = m_model_config.confidence_threshold;
    const float nms_threshold = m_nms_threshold;

    if (!input_ptr || infer_param.out_tensor_rows != (3 + 5 * 3))
    {
        std::cerr << "模型输出形状不匹配，请检查模型路径" << std::endl;
        return {};
    }

    std::vector<int> class_ids_temp;         // 符叶类别ID
    std::vector<float> confidences_temp;     // 检测置信度
    std::vector<cv::Point2d> keypoints_temp; // 符叶五个关键点坐标
    std::vector<int> indices_temp;           // NMS后保留的结果索引
    std::vector<NetRuneResult> results;      // 完整的符叶检测结果

    // 打符网络形状是 18*6300，每列是一个预测结果
    for (int candidate_index = 0; candidate_index < infer_param.out_tensor_cols; ++candidate_index)
    {
        // 封装候选项，便于后续直接当作张量计算
        const CandidateView candidate{input_ptr, static_cast<std::size_t>(candidate_index), 1,
                                      static_cast<std::size_t>(infer_param.out_tensor_cols)};

        // 选择置信度最高的符叶类别。
        int class_id = 0;
        float best_class_score = candidate[0];
        for (int class_index = 1; class_index < 3; ++class_index)
        {
            const float class_score = candidate[class_index];
            if (class_score > best_class_score)
            {
                best_class_score = class_score;
                class_id = class_index;
            }
        }

        if (best_class_score < confidence_threshold)
            continue;

        // 统计达到置信度要求的关键点数量。
        int valid_keypoint_count = 0;
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            const float keypoint_confidence = candidate[3 + point_index * 3 + 2];
            if (keypoint_confidence > 0.8f)
                ++valid_keypoint_count;
        }

        if (valid_keypoint_count < 3)
            continue;

        // 解码符叶关键点并还原到原图尺度。
        std::vector<cv::Point2d> current_keypoints(5);
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            const float x = candidate[3 + point_index * 3 + 0];
            const float y = candidate[3 + point_index * 3 + 1];

            const float kpt_x = (x - infer_param.pad_x) / infer_param.scale;
            const float kpt_y = (y - infer_param.pad_y) / infer_param.scale;

            current_keypoints[point_index] = cv::Point2d(kpt_x, kpt_y);
        }

        for (const cv::Point2d &point : current_keypoints)
            keypoints_temp.emplace_back(point);

        // 保存候选结果的各个分量。
        confidences_temp.emplace_back(best_class_score);
        class_ids_temp.emplace_back(class_id);
    }

    // 候选容器同步写入，类别为空表示没有有效结果
    if (class_ids_temp.empty())
        return {};

    // 非极大值抑制：使用的是符叶中心四点的xy平均值，即符叶中心点坐标的距离代替iou，nms阈值为30-100左右（原图像素尺寸）
    centerDistanceNMS(keypoints_temp, confidences_temp, class_ids_temp, nms_threshold, indices_temp);

    // 遍历NMS保留索引并生成最终结果
    results.reserve(indices_temp.size());
    for (const int &index : indices_temp)
    {
        // 准备结果
        NetRuneResult result;

        // 检查并存储关键点
        bool throw_flag = false;
        for (int point_index = 0; point_index < 5; ++point_index)
        {
            // 检查坐标是否超出模型输入范围，防止后续计算出错
            if (keypoints_temp[index * 5 + point_index].x > infer_param.origin_width ||
                keypoints_temp[index * 5 + point_index].x < 0 ||
                keypoints_temp[index * 5 + point_index].y > infer_param.origin_height ||
                keypoints_temp[index * 5 + point_index].y < 0)
            {
                throw_flag = true;
                break;
            }
        }
        if (throw_flag)
            continue; // 坐标无效，跳过此检测框

        result.points = {
            cv::Point2d(keypoints_temp[index * 5 + 0].x, keypoints_temp[index * 5 + 0].y),
            cv::Point2d(keypoints_temp[index * 5 + 1].x, keypoints_temp[index * 5 + 1].y),
            cv::Point2d(keypoints_temp[index * 5 + 2].x, keypoints_temp[index * 5 + 2].y),
            cv::Point2d(keypoints_temp[index * 5 + 3].x, keypoints_temp[index * 5 + 3].y),
            cv::Point2d(keypoints_temp[index * 5 + 4].x, keypoints_temp[index * 5 + 4].y)};
        result.top = result.points[0];
        result.left = result.points[1];
        result.point_R = result.points[2];
        result.right = result.points[3];
        result.bottom = result.points[4];

        // 存储类别
        result.class_id = class_ids_temp[index];
        result.score = confidences_temp[index];
        result.class_name = m_class_names[result.class_id];

        // 相当于push_back，但是更省空间，不用拷贝
        results.emplace_back(std::move(result));
    }

    return results;
}
