cmake_minimum_required(VERSION 3.16)
project(vision_training LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)

# Ceres 可选：未安装时任务2改用 Eigen 最小二乘拟合（讲义说明"拟合方法不限"）
find_package(Ceres QUIET)
# Ceres 旧版导出目标名为 `ceres`，新版为 `Ceres::ceres`，两者都兼容
set(CERES_LIBS "")
set(HAVE_CERES OFF)
if(TARGET Ceres::ceres)
    set(CERES_LIBS Ceres::ceres)
    set(HAVE_CERES ON)
elseif(TARGET ceres)
    set(CERES_LIBS ceres)
    set(HAVE_CERES ON)
endif()
if(HAVE_CERES)
    add_compile_definitions(HAVE_CERES)
    message(STATUS "Ceres found -> 任务2可用 Ceres 拟合")
else()
    message(STATUS "Ceres NOT found -> 任务2改用 Eigen 最小二乘")
endif()

# ---- 公共工具库（三个任务共用） ----
add_library(common
    src/common/image_utils.cpp
    src/common/video_io.cpp
    src/common/detection.cpp
    src/common/yolo.cpp
)
target_include_directories(common PUBLIC include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(common PUBLIC ${OpenCV_LIBS} Eigen3::Eigen ${CERES_LIBS})

# ---- 三个任务可执行程序 ----
add_executable(task1 src/task1_image/main.cpp)
target_link_libraries(task1 PRIVATE common)

add_executable(task2 src/task2_fit/main.cpp)
target_link_libraries(task2 PRIVATE common)

add_executable(task3 src/task3_windmill/main.cpp)
target_link_libraries(task3 PRIVATE common)
