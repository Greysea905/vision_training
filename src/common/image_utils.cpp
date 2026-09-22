#include "common/image_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <filesystem>

namespace common {

    /* 创建文件夹并保存图像 */
bool saveImage(const cv::Mat& img, const std::string& path) {
    if (img.empty()) return false;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    return cv::imwrite(path, img);
}

/* 计算目标点相对于已知旋转中心的极角  */
double angleFromCenter(const cv::Point2d& c, const cv::Point2d& p) { 
    return std::atan2(c.y - p.y, p.x - c.x);
}

/* 解缠角度 */
void unwrapAngles(std::vector<double>& angles) {
    if (angles.size() < 2) return;
    const double PI = std::acos(-1.0);
    const double TWO_PI = 2.0 * PI;
    double offset = 0.0;         // 累计的 2π 修正量
    double prev_raw = angles[0]; // 上一个「原始」角度（未解缠），避免用已改写的值
    for (size_t i = 1; i < angles.size(); ++i) {
        double d = angles[i] - prev_raw;   // 两个 raw 之差，必在 (−2π, 2π) 内
        if (d >  PI) offset -= TWO_PI;     // −π → +π 的 wrap（顺时针）
        if (d < -PI) offset += TWO_PI;     // +π → −π 的 wrap（逆时针）
        prev_raw = angles[i];              // 先记住 raw，再改写
        angles[i] += offset;               // raw + 修正 = 解缠值
    }
}

}  // namespace common
