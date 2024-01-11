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
#include <armor_detectors/BaseArmorDetector.hpp>
#include <armor_detectors/NetArmorDetector.hpp>
#include <armor_detectors/TraditionalArmorDetector.hpp>
#include <autoaim_interfaces/msg/detail/debug_armors__struct.hpp>
#include <autoaim_interfaces/msg/detail/debug_lights__struct.hpp>
#include <cmath>
#include <cstddef>
#include <cv_bridge/cv_bridge.h>
#include <geometry_msgs/msg/detail/point__struct.hpp>
#include <geometry_msgs/msg/detail/transform_stamped__struct.hpp>
#include <image_transport/image_transport.hpp>
#include <math.h>
#include <memory>
#include <opencv2/calib3d.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <rclcpp/duration.hpp>
#include <rclcpp/logging.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tuple>

namespace helios_cv {
    
DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : rclcpp::Node("detector_node", options) {
    // create params
    try {
        param_listener_ = std::make_shared<ParamListener>(this->get_node_parameters_interface());
        params_ = param_listener_->get_params();
    } catch (const std::exception &e) {
        RCLCPP_FATAL(logger_, "Failed to get parameters: %s, use empty params", e.what());
    }
    // init detectors
    init_detectors();
    // init debug info
    if (params_.debug) {
        binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
        result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
        number_img_pub_ = image_transport::create_publisher(this, "/detector/number_img");
        lights_data_pub_ =
            this->create_publisher<autoaim_interfaces::msg::DebugLights>("/detector/debug_lights", 10);
        armors_data_pub_ =
            this->create_publisher<autoaim_interfaces::msg::DebugArmors>("/detector/debug_armors", 10);
    }
    // create publishers and subscribers
    // create publishers
    armors_pub_ = this->create_publisher<autoaim_interfaces::msg::Armors>("/detector/armors", 10);
    // create cam info subscriber
    cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera_info", rclcpp::SensorDataQoS(),
        [this](sensor_msgs::msg::CameraInfo::SharedPtr camera_info) {
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        pnp_solver_ = std::make_shared<PnPSolver>(cam_info_->k, camera_info->d, PnPParams{
            params_.pnp_solver.small_armor_width,
            params_.pnp_solver.small_armor_height,
            params_.pnp_solver.large_armor_width,
            params_.pnp_solver.large_armor_height,
            params_.pnp_solver.energy_armor_width,
            params_.pnp_solver.energy_armor_height
        });
        project_yaw_ = std::make_shared<ProjectYaw>(cam_info_->k, camera_info->d);
        armor_detector_->set_cam_info(camera_info);
        energy_detector_->set_cam_info(camera_info);
        cam_info_sub_.reset();
    });
    // init tf2 utilities
    tf2_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    // Create the timer interface before call to waitForTransform,
    // to avoid a tf2_ros::CreateTimerInterfaceException exception
    auto timer_interface = std::make_shared<tf2_ros::CreateTimerROS>(this->get_node_base_interface(), this->get_node_timers_interface());
    tf2_buffer_->setCreateTimerInterface(timer_interface);
    tf2_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf2_buffer_);
    // subscriber and filter    
    image_sub_.subscribe(this, "/image_raw", rmw_qos_profile_sensor_data);
    // Register a callback with tf2_ros::MessageFilter to be called when transforms are available
    tf2_filter_ = std::make_shared<tf2_filter>(
        image_sub_, *tf2_buffer_, "camera_optical_frame", 10, this->get_node_logging_interface(),
        this->get_node_clock_interface(), std::chrono::duration<int>(2));
    if (params_.autoaim_mode == 0) {
        tf2_filter_->registerCallback(&DetectorNode::armor_image_callback, this);
    } else {
        tf2_filter_->registerCallback(&DetectorNode::energy_image_callback, this);
    }
    std::thread([this]()->void {
        while(rclcpp::ok()) {
            if (params_.autoaim_mode != last_autoaim_mode_) {
                if (params_.autoaim_mode == 0) {
                    RCLCPP_WARN(logger_, "Change state to armor mode");
                    // reset to release running callback function
                    tf2_filter_.reset();
                    tf2_filter_ = std::make_shared<tf2_filter>(
                        image_sub_, *tf2_buffer_, "camera_optical_frame", 10, this->get_node_logging_interface(),
                        this->get_node_clock_interface(), std::chrono::duration<int>(2));
                    tf2_filter_->registerCallback(&DetectorNode::armor_image_callback, this);   
                    last_autoaim_mode_ = 0;     
                } else {
                    RCLCPP_WARN(logger_, "Change state to energy mode");
                    // reset to release running callback function
                    tf2_filter_.reset();
                    tf2_filter_ = std::make_shared<tf2_filter>(
                        image_sub_, *tf2_buffer_, "camera_optical_frame", 10, this->get_node_logging_interface(),
                        this->get_node_clock_interface(), std::chrono::duration<int>(2));
                    tf2_filter_->registerCallback(&DetectorNode::energy_image_callback, this);        
                    last_autoaim_mode_ = params_.autoaim_mode;
                }
            }
        }
    }).detach();
}

void DetectorNode::init_detectors() {
    // create detectors
    if (params_.use_traditional) {
        // pass traditional armor detector params
        armor_detector_ = std::make_shared<TraditionalArmorDetector>(
            TAParams{
                BaseArmorParams{
                    static_cast<bool>(params_.is_blue),
                    static_cast<bool>(params_.autoaim_mode),
                    params_.debug,
                    static_cast<bool>(params_.use_traditional),
                },
                static_cast<int>(params_.armor_detector.traditional.binary_thres),
                params_.armor_detector.traditional.number_classifier_threshold,
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
                }
            }
        );
        energy_detector_ = std::make_shared<TraditionalEnergyDetector>(
            TEParams{
                BaseEnergyParam{
                    static_cast<bool>(params_.is_blue),
                    static_cast<bool>(params_.autoaim_mode),
                    params_.debug,
                    static_cast<bool>(params_.use_traditional),
                },
                static_cast<int>(params_.energy_detector.binary_thres),
                static_cast<int>(params_.energy_detector.energy_thresh),
                TEParams::RGBWeightParam{
                    params_.energy_detector.rgb_weight_r_1,
                    params_.energy_detector.rgb_weight_r_2,
                    params_.energy_detector.rgb_weight_r_3,
                    params_.energy_detector.rgb_weight_b_1,
                    params_.energy_detector.rgb_weight_b_2,
                    params_.energy_detector.rgb_weight_b_3,
                },
                params_.energy_detector.area_ratio
            }
        );
    } else {
        // pass net armor detector params
        armor_detector_ = std::make_shared<NetArmorDetector>(
            NAParams{
                BaseArmorParams{
                    static_cast<bool>(params_.is_blue),
                    static_cast<bool>(params_.autoaim_mode),
                    params_.debug,
                    static_cast<bool>(params_.use_traditional),
                },
                static_cast<int>(params_.armor_detector.net.classifier_thresh),
            }
        );
        ///TODO: pass energy detector params


    }
    armor_detector_->init();
    energy_detector_->init();
}


void DetectorNode::armor_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg) {
    if (param_listener_->is_old(params_)) {
        // need imporve: apply changed param to detector
        params_ = param_listener_->get_params();
        RCLCPP_INFO(logger_, "Params updated");
        update_detector_params();
    }
    // convert image msg to cv::Mat
    try {
        image_ = std::move(cv_bridge::toCvShare(image_msg, sensor_msgs::image_encodings::RGB8)->image);
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(logger_, "cv_bridge exception: %s", e.what());
        return;
    }
    // detect
    auto armors = armor_detector_->detect(image_);
    autoaim_interfaces::msg::Armor temp_armor;
    if (pnp_solver_ == nullptr) {
        RCLCPP_WARN(logger_, "Camera info not received, skip");
        return;
    }
    armors_msg_.armors.clear();
    armors_msg_.header = image_msg->header;
    geometry_msgs::msg::TransformStamped ts_odom2cam, ts_cam2odom;
    try {
        ts_odom2cam = tf2_buffer_->lookupTransform("camera_optical_frame", "odom", image_msg->header.stamp, 
            rclcpp::Duration::from_seconds(0.01));
        ts_cam2odom = tf2_buffer_->lookupTransform("odom", "camera_optical_frame", image_msg->header.stamp, 
            rclcpp::Duration::from_seconds(0.01));
    } catch (const tf2::TransformException & ex) {
        RCLCPP_ERROR_ONCE(get_logger(), "Error while transforming %s", ex.what());
        return;
    }
    if (project_yaw_ != nullptr) {
        // quaternion to rotation matrix
        project_yaw_->set_odom2cam(ts_odom2cam);
        project_yaw_->set_cam2odom(ts_cam2odom);
    }
    for (const auto & armor : armors) {
        cv::Mat rvec, tvec, rotation_matrix;
        bool success = pnp_solver_->solvePnP(armor, rvec, tvec);
        if (success) {
            // Fill basic info  
            temp_armor.type = static_cast<int>(armor.type);
            temp_armor.number = armor.number;
            // Fill pose
            temp_armor.pose.position.x = tvec.at<double>(0);
            temp_armor.pose.position.y = tvec.at<double>(1);
            temp_armor.pose.position.z = tvec.at<double>(2);
            // if project yaw is in error mode, use pnp solver
            if (project_yaw_ != nullptr) {
                // rvec to 3x3 rotation matrix
                cv::Mat armor_pose_in_cam;
                cv::Rodrigues(rvec, armor_pose_in_cam);
                project_yaw_->caculate_armor_yaw(armor, armor_pose_in_cam, tvec);
                tf2::Matrix3x3 tf2_rotation_matrix(
                armor_pose_in_cam.at<double>(0, 0), armor_pose_in_cam.at<double>(0, 1),
                armor_pose_in_cam.at<double>(0, 2), armor_pose_in_cam.at<double>(1, 0),
                armor_pose_in_cam.at<double>(1, 1), armor_pose_in_cam.at<double>(1, 2),
                armor_pose_in_cam.at<double>(2, 0), armor_pose_in_cam.at<double>(2, 1),
                armor_pose_in_cam.at<double>(2, 2));
                tf2::Quaternion tf2_q;
                tf2_rotation_matrix.getRotation(tf2_q);
                temp_armor.pose.orientation = tf2::toMsg(tf2_q);
            } else {
                // rvec to 3x3 rotation matrix
                cv::Rodrigues(rvec, rotation_matrix);
                // rotation matrix to quaternion
                tf2::Matrix3x3 tf2_rotation_matrix(
                rotation_matrix.at<double>(0, 0), rotation_matrix.at<double>(0, 1),
                rotation_matrix.at<double>(0, 2), rotation_matrix.at<double>(1, 0),
                rotation_matrix.at<double>(1, 1), rotation_matrix.at<double>(1, 2),
                rotation_matrix.at<double>(2, 0), rotation_matrix.at<double>(2, 1),
                rotation_matrix.at<double>(2, 2));
                tf2::Quaternion tf2_q;
                tf2_rotation_matrix.getRotation(tf2_q);
                temp_armor.pose.orientation = tf2::toMsg(tf2_q);
            }
            // Fill the distance to image center
            temp_armor.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);
            armors_msg_.armors.emplace_back(temp_armor);
        }
    }
    // publish
    armors_pub_->publish(armors_msg_);
    // debug info
    if (params_.debug) {
        publish_debug_infos();
    }
}

void DetectorNode::energy_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg) {
    ///TODO: need improve : we should restruct the energy detector and predictor with a clear line
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_INFO(logger_, "Params updated");
        update_detector_params();
    }
    // convert image msg to cv::Mat
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(image_msg, sensor_msgs::image_encodings::BGR8);
    } catch (const cv_bridge::Exception &e) {
        RCLCPP_ERROR(logger_, "cv_bridge exception: %s", e.what());
        return;
    }
    auto armors = energy_detector_->detect(cv_ptr->image);
    armors.header = image_msg->header;
    // publish
    armors_pub_->publish(armors);
    // debug info
    if (params_.debug) {
        publish_debug_infos();
    }
}

void DetectorNode::publish_debug_infos() {
    if (params_.autoaim_mode == 0) {
        // get debug infos
        const autoaim_interfaces::msg::DebugArmors* debug_armors_msg;
        const autoaim_interfaces::msg::DebugLights* debug_lights_msg;
        const cv::Mat *binary_img, *result_img, *number_img;
        std::tie(debug_lights_msg, debug_armors_msg) = armor_detector_->get_debug_msgs();
        std::tie(binary_img, result_img, number_img) = armor_detector_->get_debug_images();
        if (result_img != nullptr) {
        auto result_img_final = result_img->clone();
            // draw project yaw
            project_yaw_->draw_projection_points(result_img_final);
            result_img_pub_.publish(cv_bridge::CvImage(armors_msg_.header, sensor_msgs::image_encodings::RGB8, result_img_final).toImageMsg());
        }
        if (binary_img != nullptr) 
            binary_img_pub_.publish(cv_bridge::CvImage(armors_msg_.header, sensor_msgs::image_encodings::MONO8, *binary_img).toImageMsg());
        if (number_img != nullptr)
            number_img_pub_.publish(cv_bridge::CvImage(armors_msg_.header, sensor_msgs::image_encodings::MONO8, *number_img).toImageMsg());
        // Publish debug armors and light
        armors_data_pub_->publish(*debug_armors_msg);
        lights_data_pub_->publish(*debug_lights_msg);
    }   
}

DetectorNode::~DetectorNode() {
    binary_img_pub_.shutdown();
    result_img_pub_.shutdown();
    armor_detector_.reset();
    energy_detector_.reset();
    pnp_solver_.reset();
    param_listener_.reset();
    RCLCPP_INFO(logger_, "DetectorNode destructed");
}

void DetectorNode::update_detector_params() {
    // clear the detector and claim a new one
    if (params_.autoaim_mode == 0) {
        armor_detector_.reset();
        if (params_.use_traditional) {
            armor_detector_ = std::make_shared<TraditionalArmorDetector>(
                TAParams{
                    BaseArmorParams{
                        static_cast<bool>(params_.is_blue),
                        static_cast<bool>(params_.autoaim_mode),
                        params_.debug,
                        static_cast<bool>(params_.use_traditional),
                    },
                    static_cast<int>(params_.armor_detector.traditional.binary_thres),
                    params_.armor_detector.traditional.number_classifier_threshold,
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
                    }
                }
            );
        } else {
            armor_detector_ = std::make_shared<NetArmorDetector>(
                NAParams{
                    BaseArmorParams{
                        static_cast<bool>(params_.is_blue),
                        static_cast<bool>(params_.autoaim_mode),
                        params_.debug,
                        static_cast<bool>(params_.use_traditional),
                    },
                    static_cast<int>(params_.armor_detector.net.classifier_thresh),
                }
            );
        }
        armor_detector_->init();
    } else {
        energy_detector_.reset();
        if (params_.use_traditional) {
            energy_detector_ = std::make_shared<TraditionalEnergyDetector>(
                TEParams{
                    BaseEnergyParam{
                        static_cast<bool>(params_.is_blue),
                        static_cast<bool>(params_.autoaim_mode),
                        params_.debug,
                        static_cast<bool>(params_.use_traditional),
                    },
                    static_cast<int>(params_.energy_detector.binary_thres),
                    static_cast<int>(params_.energy_detector.energy_thresh),
                    TEParams::RGBWeightParam{
                        params_.energy_detector.rgb_weight_r_1,
                        params_.energy_detector.rgb_weight_r_2,
                        params_.energy_detector.rgb_weight_r_3,
                        params_.energy_detector.rgb_weight_b_1,
                        params_.energy_detector.rgb_weight_b_2,
                        params_.energy_detector.rgb_weight_b_3,
                    },
                    params_.energy_detector.area_ratio
                }
            );
        } else {
            ///TODO: network energy detector

        }
        energy_detector_->init();
    }
}

} // namespace helios_cv

// register node to component
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::DetectorNode);