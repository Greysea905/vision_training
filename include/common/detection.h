#pragma once

#include <opencv2/core.hpp>

namespace common {

// 由 BGR 图按两段 HSV 区间生成二值掩膜（用于跨 H 两端的颜色，如红色/青色）。
// 只需单区间时，把 low2/high2 传成与 low1/high1 相同即可。
cv::Mat colorMask(const cv::Mat& bgr,
                  const cv::Scalar& low1, const cv::Scalar& high1,
                  const cv::Scalar& low2, const cv::Scalar& high2);

}  // namespace common
