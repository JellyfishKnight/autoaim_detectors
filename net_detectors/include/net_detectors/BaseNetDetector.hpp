// created by liuhan, lijunqi on 2024/2/1
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

#include <autoaim_utilities/ThreadPool.hpp>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <rclcpp/rclcpp.hpp>

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <opencv2/core.hpp>
#include <utility>

#include "autoaim_utilities/ThreadPool.hpp"
#include "autoaim_utilities/Armor.hpp"

namespace helios_cv {

typedef struct NetParams {
    explicit NetParams(
        const std::string& model_path,
        int input_w,
        int input_h,
        int num_class,
        int num_colors,
        float nms_thresh,
        int num_apex,
        int pool_num
    ) : MODEL_PATH(model_path),
        INPUT_W(input_w),
        INPUT_H(input_h),
        NUM_CLASS(num_class),
        NUM_COLORS(num_colors),
        NMS_THRESH(nms_thresh),
        NUM_APEX(num_apex),
        POOL_NUM(pool_num) {}
    std::string MODEL_PATH;
    int INPUT_W;//输入图片的宽 416
    int INPUT_H;//输入图片的高 416
    int NUM_CLASS;//类别总数 9
    int NUM_COLORS;//颜色 2
    float NMS_THRESH;//NMS阈值 0.2
    int NUM_APEX; // 4 
    int POOL_NUM; // Thread number 3
}NetParams;

typedef struct BaseNetDetectorParams {
    bool is_blue;
    bool autoaim_mode;
    bool debug;
    double classifier_threshold;
}BaseNetDetectorParams;

class BaseNetDetector {
public:
    BaseNetDetector(const NetParams& net_params) : net_params_(net_params) {}

    ~BaseNetDetector() = default;

    virtual std::vector<Armor> detect_armors(const std::pair<rclcpp::Time, cv::Mat> image_stamped) = 0;

    virtual void set_params(void* params) = 0;

    virtual void set_cam_info(const sensor_msgs::msg::CameraInfo& cam_info) {
        cam_info_ = cam_info;
        cam_center_ = cv::Point2f(cam_info_.width / 2.0, cam_info_.height / 2.0);
    };

    virtual std::map<std::string, const cv::Mat*> get_debug_images() = 0;

protected:
    sensor_msgs::msg::CameraInfo cam_info_;
    cv::Point2f cam_center_;

    NetParams net_params_;

    std::shared_ptr<ThreadPool> thread_pool_;
};


} // namespace helios_cv