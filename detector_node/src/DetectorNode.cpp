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
#include <ament_index_cpp/get_package_share_directory.hpp>
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
#include <net_detectors/BaseNetDetector.hpp>
#include <net_detectors/NetDetectors.hpp>
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
#include <traditional_detectors/BaseTraditionalDetector.hpp>
#include <traditional_detectors/TraditionalArmorDetector.hpp>
#include <traditional_detectors/TraditionalEnergyDetector.hpp>
#include <tuple>
#include <vector>

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
        net_detector_->set_cam_info(*camera_info);
        traditional_detector_->set_cam_info(*camera_info);
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
    net_detector_ = std::make_shared<OVNetDetector>(
        BaseNetDetectorParams{
            static_cast<bool>(params_.is_blue),
            static_cast<bool>(params_.autoaim_mode),
            params_.debug,
            params_.net.classifier_thresh,
            params_.traditional.armor_detector.armor.min_large_center_distance,
            BaseNetDetectorParams::NetParams{
                ament_index_cpp::get_package_share_directory("net_detectors") + "/model/" + params_.net.model_name,
                static_cast<int>(params_.net.input_width),
                static_cast<int>(params_.net.input_height),
                static_cast<int>(params_.net.num_class),
                static_cast<int>(params_.net.num_color),
                static_cast<float>(params_.net.nms_thresh),
                static_cast<int>(params_.net.num_apex),
                static_cast<int>(params_.net.pool_num)
            }
        }
    );
    traditional_detector_ = std::make_shared<TraditionalArmorDetector>(
        TraditionalArmorParams{
            BaseTraditionalParams{
                static_cast<bool>(params_.is_blue),
                static_cast<bool>(params_.autoaim_mode),
                params_.debug,
                params_.traditional.armor_detector.binary_thres,
            },
            params_.traditional.armor_detector.number_classifier_threshold,
            TraditionalArmorParams::LightParams{
                params_.traditional.armor_detector.light.min_ratio,
                params_.traditional.armor_detector.light.max_ratio,
                params_.traditional.armor_detector.light.max_angle
            },
            TraditionalArmorParams::ArmorParams{
                params_.traditional.armor_detector.armor.min_light_ratio,
                params_.traditional.armor_detector.armor.min_small_center_distance,
                params_.traditional.armor_detector.armor.max_small_center_distance,
                params_.traditional.armor_detector.armor.min_large_center_distance,
                params_.traditional.armor_detector.armor.max_large_center_distance,
                params_.traditional.armor_detector.armor.max_angle
            }
        }
    );
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
    std::vector<Armor> armors;
    armors_msg_.header = image_msg->header;
    if (params_.use_traditional) {
        armors = traditional_detector_->detect_armors(image_);
    } else {
        auto armors_stamped = net_detector_->detect_armors(ImageStamped{image_msg->header.stamp, image_});
        armors = armors_stamped.armors;
        armors_msg_.header.stamp = armors_stamped.stamp;
    }
    autoaim_interfaces::msg::Armor temp_armor;
    if (pnp_solver_ == nullptr) {
        RCLCPP_WARN(logger_, "Camera info not received, skip");
        return;
    }
    armors_msg_.armors.clear();
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
        project_yaw_->odom2cam_r_ = project_yaw_->get_transform_info(ts_odom2cam);
        project_yaw_->cam2odom_r_ = project_yaw_->get_transform_info(ts_cam2odom);
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
    std::vector<Armor> armors;
    armors_msg_.header = image_msg->header;
    if (params_.use_traditional) {
        armors = traditional_detector_->detect_armors(cv_ptr->image);
    } else {
        auto armors_stamped = net_detector_->detect_armors(ImageStamped{image_msg->header.stamp, cv_ptr->image});
        armors_msg_.header.stamp = armors_stamped.stamp;
        armors = armors_stamped.armors;
    }
    // publish
    // armors_pub_->publish(armors);
    // debug info
    if (params_.debug) {
        publish_debug_infos();
    }
}

void DetectorNode::publish_debug_infos() {
    if (params_.use_traditional) {
        auto images = traditional_detector_->get_debug_images();
        if (params_.autoaim_mode == 0) {
            auto binary_img = images.find("binary_img");
            if (binary_img != images.end()) {
                binary_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, 
                                         *binary_img->second).toImageMsg());
            }
            auto result_img = images.find("result_img");
            if (result_img != images.end()) {
                result_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::RGB8,
                                         *result_img->second).toImageMsg());
            }
            auto number_img = images.find("number_img");
            if (number_img != images.end()) {
                number_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8,
                                         *number_img->second).toImageMsg());
            }
            autoaim_interfaces::msg::DebugArmors debug_armors;
            autoaim_interfaces::msg::DebugLights debug_lights;
            std::tie(debug_armors, debug_lights) = traditional_detector_->get_debug_infos();
            armors_data_pub_->publish(debug_armors);
            lights_data_pub_->publish(debug_lights);
        } else {
            auto binary_img = images.find("binary_img");
            if (binary_img != images.end()) {
                binary_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::MONO8, 
                                         *binary_img->second).toImageMsg());
            }
            auto result_img = images.find("detect_img");
            if (result_img != images.end()) {
                result_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::RGB8, 
                                         *result_img->second).toImageMsg());
            }
            auto prepro_img = images.find("prepro_img");
            if (prepro_img != images.end()) {
                number_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::RGB8,
                                         *prepro_img->second).toImageMsg());
            }
            autoaim_interfaces::msg::DebugArmors debug_armors;
            autoaim_interfaces::msg::DebugLights debug_lights;
            std::tie(debug_armors, debug_lights) = traditional_detector_->get_debug_infos();
            armors_data_pub_->publish(debug_armors);
            lights_data_pub_->publish(debug_lights);
        }
    } else {
        auto images = net_detector_->get_debug_images();
        auto result_img = images.find("result_img");
        if (result_img != images.end()) {
            result_img_pub_.publish(cv_bridge::CvImage(std_msgs::msg::Header(), sensor_msgs::image_encodings::RGB8,
                                     *result_img->second).toImageMsg());
        }
    }
}

DetectorNode::~DetectorNode() {
    binary_img_pub_.shutdown();
    result_img_pub_.shutdown();
    traditional_detector_.reset();
    net_detector_.reset();
    pnp_solver_.reset();
    param_listener_.reset();
    RCLCPP_INFO(logger_, "DetectorNode destructed");
}

void DetectorNode::update_detector_params() {
    // clear the detector and claim a new one
    TraditionalEnergyParams traditional_energy_params;
    TraditionalArmorParams traditional_armor_params;
    BaseNetDetectorParams net_params;
    traditional_energy_params.energy_thresh = params_.traditional.energy_detector.energy_thresh;
    traditional_energy_params.binary_threshold = params_.traditional.energy_detector.binary_thres;
    traditional_energy_params.area_ratio = params_.traditional.energy_detector.area_ratio;
    traditional_energy_params.debug = traditional_armor_params.debug = net_params.debug = params_.debug;
    traditional_energy_params.autoaim_mode = traditional_armor_params.autoaim_mode = net_params.autoaim_mode = params_.autoaim_mode;
    traditional_energy_params.is_blue = traditional_armor_params.is_blue = net_params.is_blue = params_.is_blue;
    traditional_energy_params.rgb_weight_b_1 = params_.traditional.energy_detector.rgb_weight_b_1;
    traditional_energy_params.rgb_weight_b_2 = params_.traditional.energy_detector.rgb_weight_b_2;
    traditional_energy_params.rgb_weight_b_3 = params_.traditional.energy_detector.rgb_weight_b_3;
    traditional_energy_params.rgb_weight_r_1 = params_.traditional.energy_detector.rgb_weight_r_1;
    traditional_energy_params.rgb_weight_r_2 = params_.traditional.energy_detector.rgb_weight_r_2;
    traditional_energy_params.rgb_weight_r_3 = params_.traditional.energy_detector.rgb_weight_r_3;
    traditional_armor_params.number_classifier_thresh = params_.traditional.armor_detector.number_classifier_threshold;
    traditional_armor_params.binary_threshold = params_.traditional.armor_detector.binary_thres;
    traditional_armor_params.light_params.min_ratio = params_.traditional.armor_detector.light.min_ratio;
    traditional_armor_params.light_params.max_ratio = params_.traditional.armor_detector.light.max_ratio;
    traditional_armor_params.light_params.max_angle = params_.traditional.armor_detector.light.max_angle;
    traditional_armor_params.armor_params.min_light_ratio = params_.traditional.armor_detector.armor.min_light_ratio;
    traditional_armor_params.armor_params.min_small_center_distance = params_.traditional.armor_detector.armor.min_small_center_distance;
    traditional_armor_params.armor_params.max_small_center_distance = params_.traditional.armor_detector.armor.max_small_center_distance;
    traditional_armor_params.armor_params.min_large_center_distance = params_.traditional.armor_detector.armor.min_large_center_distance;
    traditional_armor_params.armor_params.max_large_center_distance = params_.traditional.armor_detector.armor.max_large_center_distance;
    traditional_armor_params.armor_params.max_angle = params_.traditional.armor_detector.armor.max_angle;
    net_params.classifier_threshold = params_.net.classifier_thresh;
    net_params.min_large_center_distance = params_.traditional.armor_detector.armor.min_large_center_distance;
    net_params.net_params = BaseNetDetectorParams::NetParams{
        ament_index_cpp::get_package_share_directory("net_detectors") + "/models/" + params_.net.model_name,
        static_cast<int>(params_.net.input_width),
        static_cast<int>(params_.net.input_height),
        static_cast<int>(params_.net.num_class),
        static_cast<int>(params_.net.num_color),
        static_cast<float>(params_.net.nms_thresh),
        static_cast<int>(params_.net.num_apex),
        static_cast<int>(params_.net.pool_num)
    };
    net_detector_->set_params(&net_params);
    if (params_.autoaim_mode == 0) {
        if (last_autoaim_mode_ != 0) {
            traditional_detector_ = std::make_shared<TraditionalArmorDetector>(traditional_armor_params);
        } else {
            traditional_detector_->set_params(&traditional_armor_params);
        }
    } else {
        if (last_autoaim_mode_ != 1) {
            traditional_detector_ = std::make_shared<TraditionalEnergyDetector>(traditional_energy_params);
        } else {
            traditional_detector_->set_params(&traditional_energy_params);
        }
    }
}

} // namespace helios_cv

// register node to component
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::DetectorNode);