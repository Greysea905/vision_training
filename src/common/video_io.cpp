#include "common/video_io.h"

namespace common {

bool openVideo(cv::VideoCapture& cap, const std::string& path) {
    cap.open(path);
    return cap.isOpened();
}

cv::VideoWriter makeWriter(const std::string& path, const cv::VideoCapture& cap) {
    const int w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    const double fps = cap.get(cv::CAP_PROP_FPS);
    return cv::VideoWriter(path,
                           cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, cv::Size(w, h));
}

}  // namespace common
