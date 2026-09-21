#include "common/detection.h"

#include <opencv2/imgproc.hpp>

namespace common {

cv::Mat colorMask(const cv::Mat& bgr,
                  const cv::Scalar& low1, const cv::Scalar& high1,
                  const cv::Scalar& low2, const cv::Scalar& high2) {
    cv::Mat hsv, m1, m2, mask;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);
    cv::inRange(hsv, low1, high1, m1);
    cv::inRange(hsv, low2, high2, m2);
    cv::bitwise_or(m1, m2, mask);
    return mask;
}

}  // namespace common
