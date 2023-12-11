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
#pragma once

// ros
#include <cstdint>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <cv_bridge/cv_bridge.h>

// image transport
#include <image_transport/image_transport.hpp>
#include <image_transport/publisher.hpp>
#include <image_transport/subscriber_filter.hpp>

// opencv
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

// interfaces
#include <sensor_msgs/msg/detail/image__struct.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include "autoaim_interfaces/msg/armors.hpp"
#include "autoaim_interfaces/msg/debug_armors.hpp"
#include "autoaim_interfaces/msg/debug_lights.hpp"
#include <visualization_msgs/msg/marker_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>

// tf2
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/create_timer_ros.h>
#include <tf2_ros/message_filter.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/convert.h>
#include <message_filters/subscriber.h>

// detectors
#include "armor_detectors/NetArmorDetector.hpp"
#include "armor_detectors/TraditionalArmorDetector.hpp"
#include "energy_detectors/NetEnergyDetector.hpp"
#include "energy_detectors/TraditionalEnergyDetector.hpp"

// submodule
#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/PnPSolver.hpp"
#include "autoaim_utilities/ProjectYaw.hpp"

// auto generated by ros2 generate_parameter_library
// https://github.com/PickNikRobotics/generate_parameter_library
#include "detector_node_parameters.hpp"

namespace helios_cv {

using Params = detector_node::Params; 
using ParamListener = detector_node::ParamListener;
using tf2_filter = tf2_ros::MessageFilter<sensor_msgs::msg::Image>;

class DetectorNode : public rclcpp::Node {
public:
    DetectorNode(const rclcpp::NodeOptions& options);

    ~DetectorNode();

private:
    /**
     * @brief image message call back funtion
     * 
     * @param image_msg recieved image msg from camera node
     */
    void armor_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg);

    void energy_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg);

    void init_detectors();

    // topic utilities
    rclcpp::Publisher<autoaim_interfaces::msg::Armors>::SharedPtr armors_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // Camera info part
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr cam_info_sub_;
    std::shared_ptr<sensor_msgs::msg::CameraInfo> cam_info_;
    std::shared_ptr<PnPSolver> pnp_solver_;    
    std::shared_ptr<ProjectYaw> project_yaw_;

    // tf2 part
    // Subscriber with tf2 message_filter
    std::shared_ptr<tf2_ros::Buffer> tf2_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf2_listener_;
    message_filters::Subscriber<sensor_msgs::msg::Image> image_sub_;
    std::shared_ptr<tf2_filter> tf2_filter_;


    /*debug info*/
    // markers
    visualization_msgs::msg::Marker armor_marker_;
    visualization_msgs::msg::Marker text_marker_;
    visualization_msgs::msg::MarkerArray marker_array_;
    void init_markers();
    void publish_markers(const autoaim_interfaces::msg::Armors& armors_msg);
    // debug publishers
    image_transport::Publisher binary_img_pub_;
    image_transport::Publisher result_img_pub_;
    image_transport::Publisher number_img_pub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DebugLights>::SharedPtr lights_data_pub_;
    rclcpp::Publisher<autoaim_interfaces::msg::DebugArmors>::SharedPtr armors_data_pub_;
    void publish_debug_infos();
    // detector pointer
    std::shared_ptr<BaseArmorDetector> armor_detector_;
    std::shared_ptr<BaseEnergyDetector> energy_detector_;
    cv::Mat image_;
    // param utilities
    Params params_;
    std::shared_ptr<ParamListener> param_listener_;
    void update_detector_params();

    uint8_t last_autoaim_mode_;

    rclcpp::Logger logger_ = rclcpp::get_logger("detector_node");
};

} // namespace helios_cv

