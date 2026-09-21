#include "common/image_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <filesystem>

namespace common {

bool saveImage(const cv::Mat& img, const std::string& path) {
    if (img.empty()) return false;
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    return cv::imwrite(path, img);
}

double angleFromCenter(const cv::Point2d& c, const cv::Point2d& p) {
    return std::atan2(c.y - p.y, p.x - c.x);
}

}  // namespace common
