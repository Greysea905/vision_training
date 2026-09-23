// 任务1：OpenCV 图片处理（郁金香，30 分）
// 5 个步骤，共输出 16 张图到 result/task1_images/
//
// 步骤① 读图+颜色转换  -> gray.png, hsv_h.png, hsv_s.png, hsv_v.png
// 步骤② 滤波对比       -> mean_filter.png, gaussian_filter.png, median_filter.png
// 步骤③ 红色提取       -> red_mask.png
// 步骤④ 形态学+轮廓    -> erode.png, dilate.png, open.png, close.png, contours_boxes.png
// 步骤⑤ 绘制+变换      -> drawing.png, rotated_35deg.png, crop_top_left.png
//
// 注意：服务器无图形界面，不用 imshow，一律 imwrite 保存结果。

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// 输出目录
const string OUT = "result/task1_images/";

// 全局变量便于调参
/* 滤波 */
const Size MEAN_KSIZE(5, 5);     // 均值滤波核大小
const Size GAUSS_KSIZE(5, 5);    // 高斯滤波核大小
const int MED_KSIZE = 5;    // 中值滤波核大小
const double GAUSS_SIGMA = 1.5;    // 高斯滤波标准差

/* 红色 HSV 双区间阈值（H: 0-180, S: 0-255, V: 0-255） */
const Scalar RED_LOW1(0, 43, 46);       
const Scalar RED_HIGH1(13, 255, 255);   // 低段红阈值 [0, 10]
const Scalar RED_LOW2(170, 43, 46);     
const Scalar RED_HIGH2(179, 255, 255); // 高段红阈值 [170, 179]

/* 形态学处理 */
const Size ERODE_KSIZE(20, 20);     // 腐蚀操作形态学核大小
const Size DILATE_KSIZE(20, 20);    // 膨胀操作形态学核大小
const Size MORPH_KSIZE(20, 20);   // 开/闭运算形态学核大小
const double MIN_CONTOUR_AREA = 90000.0;     // 面积筛选阈值

// ① 读图检查 + 灰度图 + H/S/V 三通道
void step1_read_color(const Mat& img) {
    Mat gray, hsv;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    imwrite(OUT + "gray.png", gray);

    cvtColor(img, hsv, COLOR_BGR2HSV);
    vector<Mat> channels;
    split(hsv, channels);
    imwrite(OUT + "hsv_h.png", channels[0]);
    imwrite(OUT + "hsv_s.png", channels[1]);
    imwrite(OUT + "hsv_v.png", channels[2]);
}

// ② 均值/高斯/中值滤波对比（记录核尺寸与 sigma）
void step2_filter(const Mat& img) {
    Mat meanImg, gaussImg, medImg;
    blur(img, meanImg, MEAN_KSIZE);
    GaussianBlur(img, gaussImg, GAUSS_KSIZE, GAUSS_SIGMA);
    medianBlur(img, medImg, MED_KSIZE);
    imwrite(OUT + "mean_filter.png", meanImg);
    imwrite(OUT + "gaussian_filter.png", gaussImg);
    imwrite(OUT + "median_filter.png", medImg);
}

// ③ HSV 双区间红色掩膜（返回掩膜，供步骤④复用）
Mat step3_red_mask(const Mat& img) {
    Mat hsv, mask1, mask2, red_mask;
    cvtColor(img, hsv, COLOR_BGR2HSV);
    inRange(hsv, RED_LOW1, RED_HIGH1, mask1);
    inRange(hsv, RED_LOW2, RED_HIGH2, mask2);
    bitwise_or(mask1, mask2, red_mask);
    imwrite(OUT + "red_mask.png", red_mask);   // 交付图
    return red_mask;                            // 内存里传给步骤④
}

// ④-1 形态学处理（返回开运算结果，供轮廓提取用）
Mat step4_morph(const Mat& red_mask) {
    Mat eroded, dilated, opened, closed;
    Mat erode_kernel = getStructuringElement(MORPH_ELLIPSE, ERODE_KSIZE);
    Mat dilate_kernel = getStructuringElement(MORPH_ELLIPSE, DILATE_KSIZE);
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, MORPH_KSIZE);
    erode(red_mask, eroded, erode_kernel);
    dilate(red_mask, dilated, dilate_kernel);
    morphologyEx(red_mask, opened, MORPH_OPEN, kernel);
    morphologyEx(red_mask, closed, MORPH_CLOSE, kernel);
    imwrite(OUT + "erode.png", eroded);
    imwrite(OUT + "dilate.png", dilated);
    imwrite(OUT + "open.png", opened);
    imwrite(OUT + "close.png", closed);
    return closed;   // 闭运算效果最好，交给步骤④-2
}

void step4_contour(const Mat& best_mask, const Mat& img) {
    vector<vector<Point>> contours;
    Mat mask_copy = best_mask.clone();   // findContours 会修改输入，用副本
    findContours(mask_copy, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    Mat result = img.clone();            // 画在原图副本上，不污染原图

    for (size_t i = 0; i < contours.size(); ++i) {
        double area = contourArea(contours[i]);
        if (area > MIN_CONTOUR_AREA) {
            drawContours(result, contours, (int)i, Scalar(0, 255, 0), 5);

            Rect bbox = boundingRect(contours[i]);
            rectangle(result, bbox, Scalar(255, 0, 0), 5);
            
            string area_str = "Area: " + to_string(static_cast<int>(area));
            Point text_pos(bbox.x, max(bbox.y - 5, 20));

            putText(result, area_str, text_pos, 
                    FONT_HERSHEY_SIMPLEX, 2, Scalar(0, 255, 255), 3, LINE_AA);
        }
    }
    imwrite(OUT + "contours_boxes.png", result);
}

void step5_draw_transform(const Mat& img) {
    if (img.empty()) return;
    const int width = img.cols;
    const int height = img.rows;
    const Point2f center(width / 2.0f, height / 2.0f);

    // 复用同一个临时 Mat 容器，避免频繁在堆上分配大内存
    Mat out_mat;
    out_mat = img.clone();
    
    // ① 画圆：绿色，圆心位于图像中心，半径为宽度的 1/8
    circle(out_mat, center, width / 8, Scalar(0, 255, 0), 5, LINE_AA);
    // ② 画矩形：蓝色，以图像中心为左上角，宽高为原图的 1/8
    rectangle(out_mat, Rect(center.x, center.y, width / 8, height / 8), Scalar(255, 0, 0), 5);
    // ③ 写文字：红色，写在中心处
    putText(out_mat, "Hello OpenCV", Point(center.x, center.y), FONT_HERSHEY_SIMPLEX, 5.0, Scalar(0, 0, 255), 5, LINE_AA);
    imwrite(OUT + "drawing.png", out_mat);

    // 逆时针旋转 35°，缩放 1.0
    Mat rot_mat = getRotationMatrix2D(center, 35.0, 1.0);
    warpAffine(img, out_mat, rot_mat, img.size());      // 复用 out_mat
    imwrite(OUT + "rotated_35deg.png", out_mat);

    // 裁剪左上角 1/4
    Rect roi(0, 0, width / 2, height / 2);
    // img(roi) 生成指针切片视图，直接传给 imwrite，不产生任何多余变量与额外深拷贝
    imwrite(OUT + "crop_top_left.png", img(roi));
}

int main() {
    Mat img = imread("resources/test_image.jpg");
    if (img.empty()) {
        cerr << "读图失败：请确认 resources/test_image.jpg 存在\n";
        return 1;
    }

    step1_read_color(img);
    step2_filter(img);

    Mat red_mask = step3_red_mask(img);   // 步骤③ 返回红色掩膜
    Mat best_mask   = step4_morph(red_mask); // 步骤④-1 返回最优结果
    step4_contour(best_mask, img);           // 步骤④-2 用开运算结果提轮廓
    step5_draw_transform(img);

    cout << "任务1 已完成 " << OUT << "\n";
    return 0;
}
