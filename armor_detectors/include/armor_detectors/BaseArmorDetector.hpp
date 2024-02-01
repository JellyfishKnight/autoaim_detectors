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

#include <autoaim_interfaces/msg/detail/debug_armors__struct.hpp>
#include <autoaim_interfaces/msg/detail/debug_lights__struct.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <vector>
#include "autoaim_utilities/Armor.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace helios_cv {

typedef struct BaseArmorParams{
    // which means targets is blue
    bool is_blue;
    bool autoaim_mode;
    bool debug;
    bool use_traditional;
}BAParams;


class BaseArmorDetector {
public:
    BaseArmorDetector() = default;

    ~BaseArmorDetector() = default;

    virtual void init() = 0;

    virtual std::vector<Armor> detect(const cv::Mat& image) = 0;

    virtual void set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) = 0;

    virtual std::tuple<const cv::Mat*, const cv::Mat*, const cv::Mat*> get_debug_images() = 0;

    virtual std::tuple<const autoaim_interfaces::msg::DebugLights*,
                       const autoaim_interfaces::msg::DebugArmors*> get_debug_msgs() = 0;       
};


} // namespace helios_cv