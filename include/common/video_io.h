#pragma once

#include <opencv2/videoio.hpp>
#include <string>

namespace common {

// 打开视频；成功返回 true 并填充 cap，失败返回 false。
bool openVideo(cv::VideoCapture& cap, const std::string& path);

// 按输入视频的分辨率与帧率创建输出 writer（mp4v 编码）。
// 返回后调用 writer.isOpened() 判断是否创建成功。
cv::VideoWriter makeWriter(const std::string& path, const cv::VideoCapture& cap);

}  // namespace common
