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

#include "BaseNetDetector.hpp"
#include <rclcpp/logger.hpp>


namespace helios_cv {

#ifdef __x86_64__
class OVNetDetector : public BaseNetDetector {
public:
    OVNetDetector(const BaseNetDetectorParams& params);

    ~OVNetDetector();

    ArmorsStamped detect_armors(const ImageStamped& image_stamped) final;

    void set_params(void* params) final;

    std::map<std::string, const cv::Mat*> get_debug_images() final;
private:
    void clear_queue(std::queue<std::future<ArmorsStamped>>& futs);

    ArmorsStamped armors_stamped_;
    ImageStamped image_stamped_;

    int frames_;

    Inference* infer1_;
    Inference* infer2_;
    Inference* infer3_;

    std::vector<Inference*> detect_vector_;
    std::queue<std::future<ArmorsStamped>> futs_;

    BaseNetDetectorParams params_;

    rclcpp::Logger logger_ = rclcpp::get_logger("OVNetDetector");
};

#elif  __aarch64__

#endif



} // namespace helios_cv