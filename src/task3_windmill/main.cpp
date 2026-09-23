// 任务3：真实能量机关视频识别与稳定跟踪
//
// 用 YOLO ONNX 检测 R标(class0) 与目标靶(class1)，叠加可视化并做稳定锁定。
// 跟踪用「角速度预测」：目标绕 R标 匀速转动，估计角速度 ω，匹配与丢失外推都用预测位置。
//
// 用法：
//   ./build/task3 <视频路径> [模型路径] [输出路径]
//   默认模型 model/best.onnx，输出 result/task3_windmill/<视频名>/recognition_overlay.mp4

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

#include "common/video_io.h"
#include "common/yolo.h"

// 可调参数：
const int    IMGSZ         = 1280;   // 必须与训练/导出一致，勿改
const float  CONF_THRESH   = 0.5f;   // 检测置信度阈值
const float  NMS_THRESH    = 0.45f;  // NMS IoU 阈值
const float  ASSOC_THRESH  = 60.0f;  // 匹配距离阈值(px)：检测到预测位置多近算同一目标
const int    LOST_TOLERANCE = 15;    // 丢失容忍帧数，超过才释放（与 config 一致）
const float  SMOOTH_PRED_W  = 0.6f;  // 位置融合时预测的权重（越大越平滑、越小越跟手）


// 当前锁定的目标
struct TrackedTarget {
    bool has = false;
    int id = 0;
    cv::Point2f center;         // 当前显示位置（平滑/预测）
    float radius = 0.f;         // 目标框半径（画圆轮廓用）
    float orbit_radius = 0.f;   // 绕 R标 的轨道半径
    float angle = 0.f;          // 最近一次检测到的角度
    float omega = 0.f;          // 角速度（rad/frame）
    bool has_angle = false;     // 是否有角度历史
    bool has_omega = false;     // 是否有角速度估计
    int last_seen_frame = -1;   // 最近一次检测到的帧号
    int lost_count = 0;         // 连续丢失帧数
};

static cv::Point2f centerOf(const common::Detection& d) {
    return cv::Point2f(d.box.x + d.box.width / 2, d.box.y + d.box.height / 2);
}

static float wrapPi(float a) {
    while (a > CV_PI) a -= 2 * CV_PI;
    while (a < -CV_PI) a += 2 * CV_PI;
    return a;
}

// 用角速度预测目标在 frame 帧时的位置（沿绕 R标 的圆周）
static cv::Point2f predict(const TrackedTarget& t, const cv::Point2f& r, int frame) {
    const float a = t.angle + t.omega * (frame - t.last_seen_frame);
    return cv::Point2f(r.x + t.orbit_radius * std::cos(a),
                       r.y + t.orbit_radius * std::sin(a));
}

// 稳定锁定状态机（采取角速度预测）
static void updateTrack(TrackedTarget& t, const std::vector<common::Detection>& targets,
                        bool has_r, const cv::Point2f& r_center, int frame_idx, int& next_id) {
    if (!t.has) {
        // 首次出现有效目标：任选一个，分配新身份
        if (!targets.empty()) {
            t.has = true;
            t.id = next_id++;
            const cv::Point2f c = centerOf(targets[0]);
            t.center = c;
            t.radius = std::max(targets[0].box.width, targets[0].box.height) / 2.f;
            t.lost_count = 0;
            t.has_angle = t.has_omega = false;
            if (has_r) {
                t.orbit_radius = cv::norm(c - r_center);
                t.angle = std::atan2(c.y - r_center.y, c.x - r_center.x);
                t.has_angle = true;
                t.last_seen_frame = frame_idx;
            }
        }
        return;
    }

    // 预测当前位置（优先用角速度预测，否则退回当前显示位置）
    cv::Point2f pred = t.center;
    if (has_r && t.has_angle) pred = predict(t, r_center, frame_idx);

    // 最近邻匹配（用预测位置，而非冻结位置）
    float best_d = 1e9f;
    int best_i = -1;
    for (size_t i = 0; i < targets.size(); ++i) {
        const float d = cv::norm(centerOf(targets[i]) - pred);
        if (d < best_d) { best_d = d; best_i = static_cast<int>(i); }
    }

    if (best_i >= 0 && best_d < ASSOC_THRESH) {
        const cv::Point2f c = centerOf(targets[best_i]);
        // 更新角度 + 角速度估计
        if (has_r) {
            const float a = std::atan2(c.y - r_center.y, c.x - r_center.x);
            const float new_orbit = cv::norm(c - r_center);
            if (t.has_angle) {
                const float dtheta = wrapPi(a - t.angle);
                const int gap = frame_idx - t.last_seen_frame;
                if (gap > 0) {
                    const float inst = dtheta / gap;
                    if (!t.has_omega) { t.omega = inst; t.has_omega = true; }
                    else { t.omega = 0.7f * t.omega + 0.3f * inst; }  // EMA 平滑
                }
                t.orbit_radius = 0.7f * t.orbit_radius + 0.3f * new_orbit;  // 轨道半径平滑
            } else {
                t.orbit_radius = new_orbit;  // 首次直接设
            }
            t.angle = a;
            t.has_angle = true;
            t.last_seen_frame = frame_idx;
        }
        // 位置：预测与检测融合（卡尔曼式，既平滑又无滞后，替代纯低通 EMA）
        t.center = SMOOTH_PRED_W * pred + (1.f - SMOOTH_PRED_W) * c;
        // 框半径平滑，减少尺寸抖动
        const float new_r = std::max(targets[best_i].box.width, targets[best_i].box.height) / 2.f;
        t.radius = 0.7f * t.radius + 0.3f * new_r;
        t.lost_count = 0;
    } else {
        ++t.lost_count;
        // 丢失期间沿旋转路径外推，圆平滑继续转
        if (has_r && t.has_angle) t.center = predict(t, r_center, frame_idx);
        if (t.lost_count > LOST_TOLERANCE) t.has = false;  // 释放，允许重选
    }
}

// 叠加可视化
static void drawOverlay(cv::Mat& frame, bool has_r, const cv::Point2f& r_center,
                        const TrackedTarget& t) {
    const cv::Scalar blue(255, 0, 0), green(0, 255, 0), red(0, 0, 255), orange(0, 165, 255);

    // R标中心（蓝色十字）
    if (has_r) cv::drawMarker(frame, r_center, blue, cv::MARKER_CROSS, 30, 2);

    if (!t.has) return;

    const bool lost = t.lost_count > 0;
    const cv::Scalar c = lost ? orange : green;

    // 扇叶圆轮廓 + 扇叶圆中心
    cv::circle(frame, t.center, static_cast<int>(t.radius), c, 3);
    cv::circle(frame, t.center, 4, c, -1);

    // 两中心连线（红）
    if (has_r) cv::line(frame, r_center, t.center, red, 2);

    // ID + 状态
    const std::string status = lost ? "lost" : "detected";
    cv::putText(frame, "ID:" + std::to_string(t.id) + " " + status,
                t.center + cv::Point2f(12, -12), cv::FONT_HERSHEY_SIMPLEX, 0.8, c, 2);

    // 角度（目标相对 R 中心，跟随移动中心）
    if (has_r) {
        const double ang = std::atan2(t.center.y - r_center.y, t.center.x - r_center.x)
                           * 180.0 / CV_PI;
        cv::putText(frame, "angle:" + std::to_string(static_cast<int>(ang)),
                    r_center + cv::Point2f(12, -12), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(0, 255, 255), 2);
    }
}

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);  // 关闭缓冲，进度实时可见
    if (argc < 2) {
        std::printf("用法: %s <视频路径> [模型路径] [输出路径]\n", argv[0]);
        return 1;
    }
    const std::string video_path = argv[1];
    const std::string model_path = argc > 2 ? argv[2] : "model/best.onnx";

    std::string out_path;
    if (argc > 3) {
        out_path = argv[3];
    } else {
        std::string base = video_path;
        const size_t s = base.find_last_of("/\\");
        if (s != std::string::npos) base = base.substr(s + 1);
        const size_t e = base.find_last_of('.');
        if (e != std::string::npos) base = base.substr(0, e);
        std::filesystem::create_directories("result/task3_windmill/" + base);
        out_path = "result/task3_windmill/" + base + "/recognition_overlay.mp4";
    }

    common::YoloDetector det;
    if (!det.load(model_path, IMGSZ, CONF_THRESH, NMS_THRESH)) {
        std::printf("[task3] 无法加载模型: %s\n", model_path.c_str());
        return 1;
    }
    std::printf("[task3] 模型已加载: %s\n", model_path.c_str());

    cv::VideoCapture cap;
    if (!common::openVideo(cap, video_path)) {
        std::printf("[task3] 无法打开视频: %s\n", video_path.c_str());
        return 1;
    }
    cv::VideoWriter writer = common::makeWriter(out_path, cap);
    if (!writer.isOpened()) {
        std::printf("[task3] 无法创建输出: %s\n", out_path.c_str());
        return 1;
    }

    TrackedTarget track;
    int next_id = 1;
    int frame_idx = 0;

    cv::Mat frame;
    while (cap.read(frame)) {
        const auto dets = det.detect(frame);

        bool has_r = false;
        cv::Point2f r_center;
        float best_r_conf = -1.f;
        std::vector<common::Detection> targets;
        for (const auto& d : dets) {
            if (d.class_id == 0) {
                if (d.conf > best_r_conf) { best_r_conf = d.conf; r_center = centerOf(d); has_r = true; }
            } else if (d.class_id == 1) {
                targets.push_back(d);
            }
        }

        updateTrack(track, targets, has_r, r_center, frame_idx, next_id);
        drawOverlay(frame, has_r, r_center, track);
        writer.write(frame);

        ++frame_idx;
        if (frame_idx % 100 == 0) std::printf("[task3] 已处理 %d 帧\n", frame_idx);
    }

    std::printf("[task3] 完成：共 %d 帧，输出 %s\n", frame_idx, out_path.c_str());
    return 0;
}
