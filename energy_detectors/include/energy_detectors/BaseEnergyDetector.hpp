// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "autoaim_interfaces/msg/armors.hpp"

#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>

#include <vector>
namespace helios_cv {

typedef struct BaseEnergyParam {
    // which means targets is blue
    bool is_blue;
    bool is_armor_autoaim;
    bool debug;
    bool use_traditional;
}BEParam;

class BaseEnergyDetector {
public:
    BaseEnergyDetector() = default;

    ~BaseEnergyDetector() = default;

    virtual void init() = 0;

    virtual autoaim_interfaces::msg::Armors detect(const cv::Mat& images) = 0;

    virtual void draw_results(cv::Mat& img) = 0;

    virtual void set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) = 0;
};

} // namespace helios_cv