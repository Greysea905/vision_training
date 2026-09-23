#pragma once

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>
#include <vector>

namespace common {

// 单个检测结果：类别 + 置信度 + 原图坐标系下的框
struct Detection {
    int class_id = -1;    // 0=R标, 1=target
    float conf = 0.f;
    cv::Rect2f box;       // 原图坐标 (x, y, width, height)
};

// YOLO ONNX 检测器（cv::dnn 后端，零额外依赖）。
// 预处理与训练端一致：letterbox 到 imgsz×imgsz、/255 归一化、BGR→RGB。
class YoloDetector {
public:
    // 加载模型并设置推理参数。
    bool load(const std::string& onnx_path, int imgsz = 1280,
              float conf_thresh = 0.5f, float nms_thresh = 0.45f);

    // 对一张 BGR 图做检测，返回所有检测结果（在原图坐标系）。
    std::vector<Detection> detect(const cv::Mat& bgr);

private:
    cv::dnn::Net net_;
    int imgsz_ = 1280;
    float conf_thresh_ = 0.5f;
    float nms_thresh_ = 0.45f;
};

}  // namespace common
