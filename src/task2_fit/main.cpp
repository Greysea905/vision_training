// 任务2：合成旋转视频参数拟合（30 分）
// 模型：ω(t) = b + A·sin(Ωt + φ)，估计 A、b、Ω、φ
// 方法：Ω 一维网格搜索 + 线性最小二乘（Eigen）
//
// 数据流：
//   视频 → 检测青色点(x,y) → θ(atan2) → unwrap → ω(中心差分)
//        → 扫 Ω × 线性LS → A,b,Ω,φ → 验证/CSV/标注视频

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <limits>

#include "common/video_io.h"
#include "common/image_utils.h"
#include "common/detection.h"

using namespace cv;
using namespace std;

// ===== 常量（按需调整） =====
const string VIDEO_PATH = "resources/task_2.mp4";
const string OUT_DIR    = "result/task2_fit/";
const Point2d ROT_CENTER(480.0, 360.0);   // 已知旋转中心

// 青色目标 HSV 双区间（示例，需实测调整；青色不跨 H 两端，两段相同即可）
const Scalar CYAN_LOW1(80, 100, 100),  CYAN_HIGH1(100, 255, 255);
const Scalar CYAN_LOW2(80, 100, 100),  CYAN_HIGH2(100, 255, 255);

// Ω 网格范围（rad/s），据数据调整
const double OMEGA_MIN = 0.05;
const double OMEGA_MAX = 8.0;

// 视频元信息（detectTargets 里读取）
double g_fps = 60.0;
int    g_frame_count = 0;

// ===== 拟合结果 =====
struct FitResult {
    double A = 0, b = 0, Omega = 0, phi = 0;  // 估计的 4 个参数
    double rmse = 0;                          // 角速度 RMSE（rad/s）
    vector<double> omega_fit;                 // 拟合角速度曲线（与观测等长）
};

// ① 逐帧检测青色点质心；检测失败用 (NaN,NaN) 占位
vector<Point2d> detectTargets() {
    //   1. 打开视频，读 fps
    //   2. 逐帧：colorMask → findLargestBlob → 存中心（失败存 NaN）
    //   3. 统计并打印有效帧数

    cv::VideoCapture cap;
    if (!common::openVideo(cap, VIDEO_PATH)) {
        cerr << "无法打开视频：" << VIDEO_PATH << "\n";
        return {};
    }

    g_fps = cap.get(cv::CAP_PROP_FPS);    // 全局变量帧率，dt = 1/fps

    vector<Point2d> targets;    // 存储检测到的目标点
    cv::Mat frame;
    int valid = 0;

    while (cap.read(frame)) {                
        Mat mask = common::colorMask(frame, CYAN_LOW1, CYAN_HIGH1,
                                     CYAN_LOW2, CYAN_HIGH2);
        Point2d center;   // 接收目标点中心
        double area;
        if (common::findLargestBlob(mask, center, area)) {
            targets.push_back(center);           // 检测成功
            ++valid;
        } else {
            targets.push_back(Point2d(NAN, NAN));// 检测失败占位
        }
    }

    g_frame_count = static_cast<int>(targets.size());
    cout << "检测完成：" << valid << " / " << g_frame_count
         << " 帧有效，fps = " << g_fps << "\n";
    return targets;
}

// ② 由目标点序列算展开角度
vector<double> computeAngles(const vector<Point2d>& targets) {
    vector<double> angles;    // 存储展开角度
    angles.reserve(targets.size());

    // 1. 逐帧算 wrapped 角度（atan2 结果在 [−π,π]）
    for (const auto& p : targets) {
        if (std::isnan(p.x) || std::isnan(p.y)) {
            // 检测失败帧：本视频 100% 检测，不会走到这里。
            // 若处理会丢帧的视频，需在此线性插值，否则 unwrap 会出错。
            angles.push_back(NAN);
        } else {
            angles.push_back(common::angleFromCenter(ROT_CENTER, p));
        }
    }

    // 2. 解缠成连续递增序列（把跨 ±π 的跳变补 2π）
    common::unwrapAngles(angles);
    return angles;
}

// ③ 中心差分求角速度（首尾用单侧差分）
vector<double> computeAngularVelocity(const vector<double>& angles, double dt) {
    const size_t n = angles.size();
    vector<double> omega;
    if (n < 2) return omega;                     // 数据太少，返回空
    omega.reserve(n);

    omega.push_back((angles[1] - angles[0]) / dt);              // 首帧：前向差分
    for (size_t i = 1; i + 1 < n; ++i) {
        omega.push_back((angles[i + 1] - angles[i - 1]) / (2.0 * dt));  // 中心差分
    }
    omega.push_back((angles[n - 1] - angles[n - 2]) / dt);      // 末帧：后向差分

    return omega;   // 长度与 angles 相同，时间轴对齐
}

// ④ Ω 网格搜索 + 线性最小二乘
FitResult fit(const vector<double>& omega, const vector<double>& time) {
    const size_t n = omega.size();
    FitResult result;
    if (n < 3) return result;   // 至少 3 个样本才能解 3 个线性参数

    // 对给定 Ω 解一次线性最小二乘，返回 SSE；解出的 [b, A·cosφ, A·sinφ] 存进 x
    auto solveForOmega = [&](double Omega, Eigen::Vector3d& x) -> double {
        Eigen::MatrixXd M(n, 3);
        Eigen::VectorXd y(n);
        for (size_t i = 0; i < n; ++i) {
            M(i, 0) = 1.0;
            M(i, 1) = std::sin(Omega * time[i]);
            M(i, 2) = std::cos(Omega * time[i]);
            y(i) = omega[i];
        }
        x = M.colPivHouseholderQr().solve(y);   // x = [b, A·cosφ, A·sinφ]
        return (M * x - y).squaredNorm();
    };

    double best_sse = std::numeric_limits<double>::max();
    double best_Omega = 0.0;
    Eigen::Vector3d best_x = Eigen::Vector3d::Zero();

    // 1. 粗扫：步长 0.01
    for (double Omega = OMEGA_MIN; Omega <= OMEGA_MAX; Omega += 0.01) {
        Eigen::Vector3d x;
        double sse = solveForOmega(Omega, x);
        if (sse < best_sse) { best_sse = sse; best_Omega = Omega; best_x = x; }
    }

    // 2. 精扫：粗扫最优值附近 ±0.01，步长 0.001
    for (double Omega = best_Omega - 0.01; Omega <= best_Omega + 0.01; Omega += 0.001) {
        Eigen::Vector3d x;
        double sse = solveForOmega(Omega, x);
        if (sse < best_sse) { best_sse = sse; best_Omega = Omega; best_x = x; }
    }

    // 3. 反解参数
    double b         = best_x(0);               // 平均角速度
    double A_cos_phi = best_x(1);               // = A·cosφ（sin(Ωt) 的系数）
    double A_sin_phi = best_x(2);               // = A·sinφ（cos(Ωt) 的系数）
    double A         = std::sqrt(A_cos_phi * A_cos_phi + A_sin_phi * A_sin_phi);
    double phi       = std::atan2(A_sin_phi, A_cos_phi);   // 已在 [−π,π]

    // 4. 检查约束 b > A
    if (b <= A) {
        cerr << "警告：b <= A（b=" << b << ", A=" << A << "），约束不满足，请检查数据\n";
    }

    // 5. 算拟合曲线与 RMSE
    result.A = A;
    result.b = b;
    result.Omega = best_Omega;
    result.phi = phi;
    result.omega_fit.reserve(n);
    double sse_final = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double fit_val = b + A * std::sin(best_Omega * time[i] + phi);
        result.omega_fit.push_back(fit_val);
        double residual = omega[i] - fit_val;
        sse_final += residual * residual;
    }
    result.rmse = std::sqrt(sse_final / n);
    return result;
}

// ⑤ 写标注视频 tracking_overlay.mp4
void writeOverlayVideo(const vector<Point2d>& targets) {
    cv::VideoCapture cap;
    if (!common::openVideo(cap, VIDEO_PATH)) {
        cerr << "无法打开视频（写标注）\n";
        return;
    }

    cv::VideoWriter writer = common::makeWriter(OUT_DIR + "tracking_overlay.mp4", cap);
    if (!writer.isOpened()) {
        cerr << "无法创建输出视频\n";
        return;
    }

    cv::Mat frame;
    size_t frame_idx = 0;
    while (cap.read(frame)) {
        if (frame_idx < targets.size() &&
            !std::isnan(targets[frame_idx].x) && !std::isnan(targets[frame_idx].y)) {
            Point center_pt((int)ROT_CENTER.x, (int)ROT_CENTER.y);
            Point target_pt((int)targets[frame_idx].x, (int)targets[frame_idx].y);

            circle(frame, center_pt, 5, Scalar(255, 255, 255), -1);   // 旋转中心：白点
            circle(frame, target_pt, 6, Scalar(0, 255, 255), 2);      // 目标：黄色圆
            line(frame, center_pt, target_pt, Scalar(0, 255, 0), 2);  // 连线：绿色
            putText(frame, "frame " + to_string(frame_idx),
                    Point(20, 40), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        } else {
            putText(frame, "LOST", Point(20, 40), FONT_HERSHEY_SIMPLEX,
                    0.8, Scalar(0, 0, 255), 2);
        }
        writer.write(frame);   // 覆盖全部帧，不丢帧、不改帧率
        ++frame_idx;
    }

    cout << "已写出标注视频，共 " << frame_idx << " 帧\n";
}

// ⑥ 导出 CSV 供 plot.py 画图
void exportCSV(const vector<double>& time,
               const vector<double>& omega_obs,
               const vector<double>& omega_fit) {
    ofstream f(OUT_DIR + "fit_data.csv");
    if (!f.is_open()) {
        cerr << "无法写入 CSV\n";
        return;
    }

    f << "t,omega_obs,omega_fit,residual\n";
    size_t n = time.size();
    if (omega_obs.size() < n) n = omega_obs.size();
    if (omega_fit.size() < n) n = omega_fit.size();
    for (size_t i = 0; i < n; ++i) {
        f << time[i] << "," << omega_obs[i] << "," << omega_fit[i] << ","
          << (omega_obs[i] - omega_fit[i]) << "\n";
    }
    f.close();
    cout << "已导出 CSV，共 " << n << " 行\n";
}

int main() {
    vector<Point2d> targets = detectTargets();
    vector<double> angles = computeAngles(targets);

    double dt = 1.0 / g_fps;
    vector<double> omega = computeAngularVelocity(angles, dt);

    vector<double> time(omega.size());
    for (size_t i = 0; i < time.size(); ++i) time[i] = i * dt;

    FitResult result = fit(omega, time);
    cout << "A=" << result.A << "  b=" << result.b
         << "  Omega=" << result.Omega << "  phi=" << result.phi
         << "  RMSE=" << result.rmse << " rad/s\n";

    writeOverlayVideo(targets);
    exportCSV(time, omega, result.omega_fit);
    return 0;
}
