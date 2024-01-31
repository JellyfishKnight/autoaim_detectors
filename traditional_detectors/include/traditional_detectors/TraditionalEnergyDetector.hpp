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

#include "BaseTraditionalDetector.hpp"
#include "TraditionalArmorDetector.hpp"


namespace helios_cv {

typedef struct TraditionalEnergyParams : public BaseTraditionalParams {
    
}TraditionalEnergyParams;

class TraditionalEnergyDetector : public BaseTraditionalDetector {
public:
    TraditionalEnergyDetector(const TraditionalArmorParams& params);

    ~TraditionalEnergyDetector() = default;

    std::vector<Armor> detect_armors(const cv::Mat& image) final;

    void set_params(void* params) final;

    std::map<std::string, const cv::Mat*> get_debug_images() final;

    std::tuple<autoaim_interfaces::msg::DebugArmors, 
                autoaim_interfaces::msg::DebugLights> get_debug_infos() final;
private:



    rclcpp::Logger logger_ = rclcpp::get_logger("TraditionalEnergyDetector");
};
} // namespace helios_cv
