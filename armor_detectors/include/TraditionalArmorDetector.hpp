// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

// base class
#include "BaseArmorDetector.hpp"

// ros
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>
#include <ament_index_cpp/get_package_share_directory.hpp>

// interfaces
#include <sensor_msgs/msg/camera_info.hpp>
#include "autoaim_interfaces/msg/armors.hpp"

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

// tf2
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.h>

// utilities
#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/NumberClassifier.hpp"
#include "autoaim_utilities/PnPSolver.hpp"

namespace helios_cv {

typedef struct TraditionalArmorDetectorParams : public BaseArmorParams {
    int binary_thresh;
    int number_classifier_thresh;
    typedef struct LightParams {
        double min_ratio;
        double max_ratio;
        double max_angle;
    }LightParams;
    LightParams light_params;
    typedef struct ArmorParams {
        double min_light_ratio;
        double max_light_ratio;
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        double max_angle;
    }ArmorParams;
    ArmorParams armor_params;
}TAParams;

class TraditionalArmorDetector : public BaseArmorDetector {
public:
    TraditionalArmorDetector(const TAParams& params);

    void init() override;

    autoaim_interfaces::msg::Armors detect(const cv::Mat& images) override;

    void draw_results(cv::Mat& img) override;

    void set_params(const TAParams& params);

    void pack() override;

private:

    cv::Mat preprocessImage(const cv::Mat & input);

    std::vector<Light> findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img);
    
    std::vector<Armor> matchLights(const std::vector<Light> & lights);

    bool isLight(const Light & possible_light);
    
    bool containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights);
    
    ArmorType isArmor(const Light & light_1, const Light & light_2);
    // submodules
    std::shared_ptr<PnPSolver> pnp_solver_;
    std::shared_ptr<NumberClassifier> number_classifier_;
    // params
    TAParams params_;
    // camera info
    cv::Point2f cam_center_;
    sensor_msgs::msg::CameraInfo::SharedPtr cam_info_;

    std::vector<Light> lights_;
    std::vector<Armor> armors_;
    autoaim_interfaces::msg::Armors armors_interfaces_;
    // frame image
    cv::Mat frame_;
    cv::Mat binary_img_;

    void convert_armors_into_interfaces();

    rclcpp::Logger logger_ = rclcpp::get_logger("TraditionalArmorDetector");
};

} // namespace helios_cv