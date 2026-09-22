#include "common/detection.h"

#include <opencv2/imgproc.hpp>
#include <vector>

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

bool findLargestBlob(const cv::Mat& mask, cv::Point2d& center, double& area) {
    if (mask.empty()) return false;

    cv::Mat m = mask.clone();   // findContours 会修改输入，用副本
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(m, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return false;

    // 找面积最大的轮廓
    int best = 0;
    double best_area = -1.0;
    for (size_t i = 0; i < contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > best_area) { best_area = a; best = static_cast<int>(i); }
    }

    // 用矩求质心（比 boundingRect 中心更贴近真实形心）
    cv::Moments mu = cv::moments(contours[best]);
    if (mu.m00 < 1e-6) return false;
    center = cv::Point2d(mu.m10 / mu.m00, mu.m01 / mu.m00);
    area = best_area;
    return true;
}

}  // namespace common
