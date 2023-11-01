// created by liuhan on 2023/10/29
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
/*
 * ██   ██ ███████ ██      ██  ██████  ███████
 * ██   ██ ██      ██      ██ ██    ██ ██
 * ███████ █████   ██      ██ ██    ██ ███████
 * ██   ██ ██      ██      ██ ██    ██      ██
 * ██   ██ ███████ ███████ ██  ██████  ███████
 */

#include "DetectorNode.hpp"

namespace helios_cv {
    
DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : rclcpp::Node("detector_node", options) {
    // create params
    try {
        param_listener_ = std::make_shared<ParamListener>(this->get_node_parameters_interface());
        params_ = param_listener_->get_params();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(logger_, "Failed to get parameters: %s, use empty params", e.what());
    }
    // create detectors
    if (params_.use_traditional) {
        // pass traditional armor detector params
        armor_detector_ = std::make_shared<TraditionalArmorDetector>(
            TAParams{
                BaseArmorParams{
                    params_.is_blue,
                    params_.is_armor_autoaim,
                    params_.debug,
                    params_.use_traditional
                },
                static_cast<int>(params_.armor_detector.traditional.binary_thres),
                static_cast<int>(params_.armor_detector.traditional.number_classifier_threshold),
                TAParams::LightParams{
                    params_.armor_detector.traditional.light.min_ratio,
                    params_.armor_detector.traditional.light.max_ratio,
                    params_.armor_detector.traditional.light.max_angle
                },
                TAParams::ArmorParams{
                    params_.armor_detector.traditional.armor.min_light_ratio,
                    params_.armor_detector.traditional.armor.min_small_center_distance,
                    params_.armor_detector.traditional.armor.max_small_center_distance,
                    params_.armor_detector.traditional.armor.min_large_center_distance,
                    params_.armor_detector.traditional.armor.max_large_center_distance,
                    params_.armor_detector.traditional.armor.max_angle,
                },
                params_.pnp_solver.small_armor_height,
                params_.pnp_solver.small_armor_width,
                params_.pnp_solver.large_armor_height,
                params_.pnp_solver.large_armor_width
            }
        );
        ///TODO: pass energy_detector params

    } else {
        // pass net armor detector params
        armor_detector_ = std::make_shared<NetArmorDetector>(
            NAParams{
                BaseArmorParams{
                    params_.is_blue,
                    params_.is_armor_autoaim,
                    params_.debug,
                    params_.use_traditional
                },
                static_cast<int>(params_.armor_detector.net.classifier_thres),
                params_.pnp_solver.small_armor_height,
                params_.pnp_solver.small_armor_width,
                params_.pnp_solver.large_armor_height,
                params_.pnp_solver.large_armor_width,
                NAParams::ArmorParams{
                    params_.armor_detector.traditional.armor.min_light_ratio,
                    params_.armor_detector.traditional.armor.min_small_center_distance,
                    params_.armor_detector.traditional.armor.max_small_center_distance,
                    params_.armor_detector.traditional.armor.min_large_center_distance,
                    params_.armor_detector.traditional.armor.max_large_center_distance,
                    params_.armor_detector.traditional.armor.max_angle
                }
            }
        );
        ///TODO: pass energy detector params

    }
    // init markers
    
    // create publishers and subscribers

}

DetectorNode::~DetectorNode() {
    RCLCPP_INFO(logger_, "DetectorNode destructed");
}

} // namespace helios_cv

// register node to component
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::DetectorNode);