// created by liuhan on 2024/2/1
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
/*
 * ██   ██ ███████ ██      ██  ██████  ███████
 * ██   ██ ██      ██      ██ ██    ██ ██
 * ███████ █████   ██      ██ ██    ██ ███████
 * ██   ██ ██      ██      ██ ██    ██      ██
 * ██   ██ ███████ ███████ ██  ██████  ███████
 */
#pragma once


// ros2 std
#include "rclcpp/rclcpp.hpp"
#include "cv_bridge/cv_bridge.h"
// opencv
#include "opencv2/opencv.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/core.hpp"
//eigen
#include "eigen3/Eigen/Dense"
#include "eigen3/Eigen/Core"
// interfaces
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "autoaim_interfaces/msg/armor.hpp"
#include "autoaim_interfaces/msg/armors.hpp"
// utilities
#include "BaseTraditionalDetector.hpp"
#include <vector>


namespace helios_cv {

typedef struct TraditionalEnergyParams : public BaseTraditionalParams {
    double energy_thresh;
    double rgb_weight_r_1;
    double rgb_weight_r_2;
    double rgb_weight_r_3;
    double rgb_weight_b_1;
    double rgb_weight_b_2;
    double rgb_weight_b_3;
    double area_ratio;
}TraditionalEnergyParams;

class TraditionalEnergyDetector : public BaseTraditionalDetector {
public:
    TraditionalEnergyDetector(const TraditionalEnergyParams& params);

    ~TraditionalEnergyDetector() = default;

    std::vector<Armor> detect_armors(const cv::Mat& image) final;

    void set_params(void* params) final;

    std::map<std::string, const cv::Mat*> get_debug_images() final;

    std::tuple<autoaim_interfaces::msg::DebugArmors, 
                autoaim_interfaces::msg::DebugLights> get_debug_infos() final;
private:
    /**
     * @brief 画识别结果
     * @param img 原图
     */
    void draw_results(cv::Mat& img);
    /**
     * @brief 与处理
     * 
     * @param src 
     * @param isred 
     * @return cv::Mat 
     */
    cv::Mat preprocess(const cv::Mat& src, bool isred);
    /**
     * @brief 寻找流动扇页和目标装甲板
     * 
     * @param src 
     * @param contours 
     * @return true 
     * @return false 
     */
    bool find_target_flow(const cv::Mat& src, std::vector<std::vector<cv::Point>> &contours);
    /**
     * @brief 寻找能量机关中心的R标
     * 
     * @param contours 
     * @return true 
     * @return false 
     */
    bool find_target_R(std::vector<std::vector<cv::Point>> &contours);

    /**
     * @brief 设置透射变换
     * 
     * @param p 
     * @param d 
     */
    inline void setTransform(cv::Point2f p[], cv::Point2f d[]);
    /**
     * @brief 设置ROI
     * 
     * @param src 
     * @return cv::Mat 
     */
    inline cv::Mat get_Roi(const cv::Mat& src);

    /**
     * @brief 求R可能存在的点
     * 
     * @return Point2f 
     */
    inline cv::Point2f R_possible();

    /**
     * @brief 设置装甲板四个点
     * 
     * @param armor_fin 
     */
    inline void getPts(cv::RotatedRect &armor_fin);
    /**
     * @brief 从ROI上的坐标变回原图上的坐标
     * 
     * @param armor_fin 
     * @param circle_center_point 
     */
    inline void setPoint(cv::RotatedRect &armor_fin, cv::Point2f &circle_center_point);

    std::vector<cv::Point2f> pts;//目标装甲板坐标 01为长边
    cv::Point2f circle_center_point = cv::Point2f(0, 0);//圆心R的坐标
    cv::Point2f pre_point;
    // debug visiualization
    cv::Mat binary_img;
    cv::Mat detect_img;
    cv::Mat prepro_img;
    // 存放轮廓
    std::vector<std::vector<cv::Point>> contours;
    // 透射变换后把图像拆成两部分，两部分的点分别存在这里
    std::vector<cv::Point> point_left, point_right;
    // 透射变换的dst
    const cv::Point2f dst[4] = {cv::Point2f(0, 0), cv::Point2f(200, 0), cv::Point2f(200, 100), cv::Point2f(0, 100)};
    // 透射变换后把图片裁成两部分
    const cv::Rect r_left = cv::Rect(0, 0, 100, 100);
    const cv::Rect r_right = cv::Rect(100, 0, 100, 100);
    //vector<vector<Point>> target_armor;//目标装甲板
    // 扇页的两部分
    cv::RotatedRect rota_far, rota_close;
    // 最终的目标装甲板的旋转矩形
    cv::RotatedRect armor_fin;
    // 初始的roi，不敢设太小
    cv::Rect rect_roi = cv::Rect(0, 0, 1280, 1000);
    // ROI左上角的坐标，初始为0
    cv::Point2f roi_point = cv::Point2f(0, 0);

    std::vector<Armor> armors_;
    bool find_target_;

    TraditionalEnergyParams params_;

    rclcpp::Logger logger_ = rclcpp::get_logger("TraditionalEnergyDetector");
};
} // namespace helios_cv
