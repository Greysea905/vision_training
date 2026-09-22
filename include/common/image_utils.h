#pragma once

#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace common {

// 保存图像到指定路径；自动创建缺失的父目录。返回是否成功。
bool saveImage(const cv::Mat& img, const std::string& path);

// 从中心 c 指向点 p 的角度（弧度）。
// 图像坐标系：x 向右、y 向下；从右方起算，逆时针为正。
// 等价于 atan2(c.y - p.y, p.x - c.x)。
double angleFromCenter(const cv::Point2d& c, const cv::Point2d& p);

// 原地解缠角度序列：把 atan2 产生的 [−π,π] 跳变补 ±2π，得到连续序列。
// angles[0] 视为已展开，其余按相邻差修正。用于旋转目标的角度累计（任务2/3）。
void unwrapAngles(std::vector<double>& angles);

}  // namespace common
