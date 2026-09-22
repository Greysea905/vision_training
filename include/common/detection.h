#pragma once

#include <opencv2/core.hpp>

namespace common {

// 由 BGR 图按两段 HSV 区间生成二值掩膜（用于跨 H 两端的颜色，如红色/青色）。
// 只需单区间时，把 low2/high2 传成与 low1/high1 相同即可。
cv::Mat colorMask(const cv::Mat& bgr,
                  const cv::Scalar& low1, const cv::Scalar& high1,
                  const cv::Scalar& low2, const cv::Scalar& high2);

// 在二值掩膜中找面积最大的连通块，返回其质心与面积（任务2/3 检测目标中心用）。
// 返回 false 表示掩膜为空或没有前景像素。
bool findLargestBlob(const cv::Mat& mask, cv::Point2d& center, double& area);

}  // namespace common
