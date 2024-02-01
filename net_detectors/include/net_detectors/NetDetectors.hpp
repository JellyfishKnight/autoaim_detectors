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


namespace helios_cv {

class OVNetDetector : public BaseNetDetector {
public:
    ArmorsStamped detect_armors(const ImageStamped& image_stamped) final;

    void set_params(void* params) final;

    std::map<std::string, const cv::Mat*> get_debug_images() final;


private:


};

} // namespace helios_cv