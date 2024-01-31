// created by liuhan on 2023/10/30
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
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include "autoaim_interfaces/msg/debug_armors.hpp"
#include "autoaim_interfaces/msg/debug_lights.hpp"

#include <opencv2/core.hpp>

#include <autoaim_utilities/Armor.hpp>

#include <string>
#include <tuple>
#include <vector>
#include <memory>
#include <map>

namespace helios_cv {

typedef struct BaseTraditionalParams{
    bool is_blue;
    bool autoaim_mode;
    bool debug;
    double binary_threshold;

}BaseTraditionalParams;

class BaseTraditionalDetector {
public:
    BaseTraditionalDetector() = default;

    ~BaseTraditionalDetector() = default;

    virtual std::vector<Armor> detect_armors(const cv::Mat& image) = 0;
    
    virtual void set_params(void* params) = 0;

    virtual void set_cam_info(const sensor_msgs::msg::CameraInfo& cam_info) {
        cam_info_ = cam_info;
        cam_center_ = cv::Point2f(cam_info_.width / 2.0, cam_info_.height / 2.0);
    };

    virtual std::map<std::string, const cv::Mat*> get_debug_images() = 0;

    virtual std::tuple<autoaim_interfaces::msg::DebugArmors, 
                        autoaim_interfaces::msg::DebugLights> get_debug_infos() = 0;

protected:
    sensor_msgs::msg::CameraInfo cam_info_;

    cv::Mat result_img_;
    cv::Mat binary_img_;

    cv::Point2f cam_center_;

    autoaim_interfaces::msg::DebugArmors debug_armors_;
    autoaim_interfaces::msg::DebugLights debug_lights_;

    std::vector<Armor> armors_;
    std::vector<Light> lights_;
};


} // namespace helios_cv