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

#include <autoaim_utilities/NumberClassifier.hpp>

#include <memory>

namespace helios_cv {

typedef struct TraditionalArmorParams: public BaseTraditionalParams {
    double number_classifier_thresh;
    typedef struct LightParams {
        double min_ratio;
        double max_ratio;
        double max_angle;
    }LightParams;
    LightParams light_params;
    typedef struct ArmorParams {
        double min_light_ratio;
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        double max_angle;
    }ArmorParams;
    ArmorParams armor_params;
}TraditionalArmorParams;

class TraditionalArmorDetector : public BaseTraditionalDetector {
public:
    TraditionalArmorDetector(const TraditionalArmorParams& params);

    ~TraditionalArmorDetector() = default;

    std::vector<Armor> detect_armors(const cv::Mat& image) final;

    void set_params(void* params) final;

    std::map<std::string, const cv::Mat*> get_debug_images() final;

    std::tuple<autoaim_interfaces::msg::DebugArmors, 
                autoaim_interfaces::msg::DebugLights> get_debug_infos() final;
private:
    cv::Mat preprocessImage(const cv::Mat & input);

    std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);
    
    std::vector<Armor> matchLights(const std::vector<Light> & lights);

    bool isLight(const Light & possible_light);
    
    bool containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
    
    ArmorType isArmor(const Light & light_1, const Light & light_2);

    std::shared_ptr<NumberClassifier> number_classifier_;
    cv::Mat number_img_;
    void get_all_number_images();

    TraditionalArmorParams params_;

    rclcpp::Logger logger_ = rclcpp::get_logger("TraditionalArmorDetector");
};

} // namespace helios_cv