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
#include "autoaim_interfaces/msg/armors.hpp"

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

    virtual autoaim_interfaces::msg::Armors detect(const cv::Mat& images) = 0;

    virtual void pack() = 0;    

    virtual void draw_results(cv::Mat& img) = 0;
};


} // namespace helios_cv