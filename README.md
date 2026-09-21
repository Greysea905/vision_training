# vision_training —— 机甲大师第二次培训

三个任务统一在同一个工程中，用 C++ + OpenCV + CMake 完成。

- 任务1：OpenCV 图片处理（郁金香，30 分）
- 任务2：合成旋转视频参数拟合（30 分）
- 任务3：真实能量机关视频识别与稳定跟踪（40 分）

## 环境依赖

| 依赖 | 版本（本机实测） | 状态 |
|---|---|---|
| CMake | 3.22.1 | ✅ |
| g++ | 11.4.0（C++17） | ✅ |
| OpenCV（C++） | 4.10.0 | ✅ |
| Eigen3 | 已安装（头文件库） | ✅ |
| Ceres Solver | 未安装 | ⚠️ 可选，任务2可改用 Eigen 最小二乘 |
| Python3 + matplotlib | 已装但 numpy 冲突 | ⚠️ 出图前需修复 |

如需安装 Ceres（可选）：
```bash
sudo apt update && sudo apt install libceres-dev
```

## 构建与运行

```bash
# 1. 把素材放入 resources/（test_image.jpg、task_2.mp4、task_3.mp4、task_4.mp4）
mkdir -p resources

# 2. 配置并编译
cmake -S . -B build
cmake --build build -j4

# 3. 运行三个程序
./build/task1
./build/task2
./build/task3
```

## 目录结构

```
vision_training/
├── CMakeLists.txt        # 根构建，生成 task1/task2/task3 三个可执行程序
├── README.md
├── include/common/       # 公共头文件
├── src/
│   ├── common/           # 公共工具（视频读写/存图/角度/颜色检测）
│   ├── task1_image/      # 任务1
│   ├── task2_fit/        # 任务2
│   └── task3_windmill/   # 任务3
├── scripts/plot.py       # 任务2出图脚本（CSV -> PNG）
├── config/params.yaml    # 集中管理可调参数
├── resources/            # 输入素材
├── result/               # 全部输出结果（提交内容）
└── build/                # 构建产物（gitignore，不提交）
```

## 各任务待办

- [ ] 任务1：读图/颜色转换、滤波对比、红色提取、形态学+轮廓、绘制变换（共16张图）
- [ ] 任务2：识别青色目标 → 角度提取 → 拟合 ω(t)=b+A·sin(Ωt+φ) → 出图/标注视频
- [ ] 任务3：两视频识别跟踪 + 稳定锁定/丢失重选状态机

## 结果输出路径（对照讲义）

- `result/task1_images/` — 任务1 的 16 张图
- `result/task2_fit/` — 标注视频 + 3 张曲线图 + `task2_fit_result.md`
- `result/task3_windmill/task_3/` 与 `task_4/` — 两个标注视频 + 说明 `task3_tracking_result.md`
