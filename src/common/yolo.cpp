#include "common/yolo.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace common {

bool YoloDetector::load(const std::string& onnx_path, int imgsz,
                        float conf_thresh, float nms_thresh) {
    imgsz_ = imgsz;
    conf_thresh_ = conf_thresh;
    nms_thresh_ = nms_thresh;
    net_ = cv::dnn::readNetFromONNX(onnx_path);
    return !net_.empty();
}

std::vector<Detection> YoloDetector::detect(const cv::Mat& bgr) {
    std::vector<Detection> result;
    if (bgr.empty() || net_.empty()) return result;

    const int imgsz = imgsz_;
    const float W = static_cast<float>(bgr.cols);
    const float H = static_cast<float>(bgr.rows);

    // ---- 1) letterbox：等比缩放 + 居中补边到 imgsz×imgsz ----
    const float scale = std::min(imgsz / W, imgsz / H);
    const int new_w = static_cast<int>(std::round(W * scale));
    const int new_h = static_cast<int>(std::round(H * scale));
    const int pad_x = (imgsz - new_w) / 2;
    const int pad_y = (imgsz - new_h) / 2;

    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(new_w, new_h));
    cv::Mat letterboxed = cv::Mat::zeros(imgsz, imgsz, bgr.type());
    resized.copyTo(letterboxed(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // ---- 2) blob：/255 归一化 + BGR→RGB（与训练一致）----
    cv::Mat blob = cv::dnn::blobFromImage(
        letterboxed, 1.0 / 255.0, cv::Size(imgsz, imgsz), cv::Scalar(), true, false);
    net_.setInput(blob);

    // ---- 3) 前向，输出 [1, 4+nc, N] ----
    cv::Mat out = net_.forward();

    // 解析输出形状（兼容 2D/3D 两种返回形式）
    int channels = 0, N = 0;
    if (out.dims == 3) {          // [batch, channels, N]
        channels = out.size[1];
        N = out.size[2];
    } else if (out.dims == 2) {   // [channels, N]（batch 被 squeeze）
        channels = out.size[0];
        N = out.size[1];
    } else {
        return result;
    }
    const int nc = channels - 4;  // 类别数
    const float* data = reinterpret_cast<const float*>(out.data);

    // ---- 4) 遍历 N 个候选框，过滤低置信度 ----
    std::vector<cv::Rect2f> boxes;   // 1280 空间
    std::vector<float> confs;
    std::vector<int> classes;

    for (int n = 0; n < N; ++n) {
        const float cx = data[0 * N + n];
        const float cy = data[1 * N + n];
        const float bw = data[2 * N + n];
        const float bh = data[3 * N + n];

        int cls = -1;
        float conf = 0.f;
        for (int c = 0; c < nc; ++c) {
            const float s = data[(4 + c) * N + n];
            if (s > conf) { conf = s; cls = c; }
        }
        if (conf < conf_thresh_) continue;

        boxes.emplace_back(cx - bw / 2, cy - bh / 2, bw, bh);
        confs.push_back(conf);
        classes.push_back(cls);
    }

    // ---- 5) 按类别分别 NMS ----
    for (int cls = 0; cls < nc; ++cls) {
        std::vector<cv::Rect> r_int;   // 供 NMS（需 int）
        std::vector<float> c_sub;
        std::vector<int> idx;          // 指向 boxes/原 float 框
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (classes[i] != cls) continue;
            const cv::Rect2f& f = boxes[i];
            r_int.emplace_back(cv::Rect(static_cast<int>(f.x), static_cast<int>(f.y),
                                        static_cast<int>(f.width), static_cast<int>(f.height)));
            c_sub.push_back(confs[i]);
            idx.push_back(static_cast<int>(i));
        }
        if (r_int.empty()) continue;

        std::vector<int> keep;
        cv::dnn::NMSBoxes(r_int, c_sub, conf_thresh_, nms_thresh_, keep);
        for (int k : keep) {
            const cv::Rect2f& f = boxes[idx[k]];

            // ---- 6) 反 letterbox：映射回原图坐标 ----
            float ox = (f.x - pad_x) / scale;
            float oy = (f.y - pad_y) / scale;
            float ow = f.width / scale;
            float oh = f.height / scale;
            // 裁剪到原图范围内
            ox = std::max(0.f, std::min(ox, W));
            oy = std::max(0.f, std::min(oy, H));
            ow = std::min(ow, W - ox);
            oh = std::min(oh, H - oy);

            Detection d;
            d.class_id = cls;
            d.conf = c_sub[k];
            d.box = cv::Rect2f(ox, oy, ow, oh);
            result.push_back(d);
        }
    }
    return result;
}

}  // namespace common
