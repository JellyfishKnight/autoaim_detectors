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

#include <opencv2/core.hpp>
#include "autoaim_utilities/Armor.hpp"
#include "sensor_msgs/msg/camera_info.hpp"

namespace helios_cv {

typedef struct BaseArmorParams{
    // which means targets is blue
    bool is_blue;
    bool is_armor_autoaim;
    bool debug;
    bool use_traditional;
}BAParams;

class BaseArmorDetector {
public:
    BaseArmorDetector() = default;

    ~BaseArmorDetector() = default;

    virtual void init() = 0;

    virtual std::vector<Armor> detect(const cv::Mat& images) = 0;

    virtual void draw_results(cv::Mat& img) = 0;

    virtual void set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) = 0;
};


} // namespace helios_cv