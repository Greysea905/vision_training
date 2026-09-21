#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace common {

// 保存图像到指定路径；自动创建缺失的父目录。返回是否成功。
bool saveImage(const cv::Mat& img, const std::string& path);

// 从中心 c 指向点 p 的角度（弧度）。
// 图像坐标系：x 向右、y 向下；从右方起算，逆时针为正。
// 等价于 atan2(c.y - p.y, p.x - c.x)。
double angleFromCenter(const cv::Point2d& c, const cv::Point2d& p);

}  // namespace common
