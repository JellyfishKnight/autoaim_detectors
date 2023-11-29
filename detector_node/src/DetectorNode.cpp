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
#include <cv_bridge/cv_bridge.h>

namespace helios_cv {
    
DetectorNode::DetectorNode(const rclcpp::NodeOptions& options) : rclcpp::Node("detector_node", options) {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
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
        init_markers();
        binary_img_pub_ = image_transport::create_publisher(this, "/detector/binary_img");
        result_img_pub_ = image_transport::create_publisher(this, "/detector/result_img");
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
        cam_center_ = cv::Point2f(camera_info->k[2], camera_info->k[5]);
        cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
        pnp_solver_ = std::make_shared<PnPSolver>(cam_info_->k, camera_info->d, PnPParams{
            params_.pnp_solver.small_armor_width,
            params_.pnp_solver.small_armor_height,
            params_.pnp_solver.large_armor_width,
            params_.pnp_solver.large_armor_height,
            params_.pnp_solver.energy_armor_width,
            params_.pnp_solver.energy_armor_height
        });
        armor_detector_->set_cam_info(camera_info);
        energy_detector_->set_cam_info(camera_info);
        cam_info_sub_.reset();
    });
    // set different callback function
    if (params_.is_armor_autoaim) {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&DetectorNode::armor_image_callback, this, std::placeholders::_1));
    } else {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&DetectorNode::energy_image_callback, this, std::placeholders::_1));
    }
}

void DetectorNode::init_detectors() {
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
                    params_.is_blue,
                    params_.is_armor_autoaim,
                    params_.debug,
                    params_.use_traditional
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
        armor_detector_->init();
        energy_detector_->init();
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
                static_cast<int>(params_.armor_detector.net.classifier_thresh),
            }
        );
        ///TODO: pass energy detector params


        armor_detector_->init();
        energy_detector_->init();
    }
}


void DetectorNode::armor_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg) {
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_INFO(logger_, "Params updated");
    }
    if (!params_.is_armor_autoaim) {
        RCLCPP_WARN(logger_, "change state to energy!");
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(), 
            std::bind(&DetectorNode::energy_image_callback, this, std::placeholders::_1));
        return ;
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
    autoaim_interfaces::msg::Armors armors_msg;
    if (pnp_solver_ == nullptr) {
        RCLCPP_WARN(logger_, "Camera info not received, skip pnp solve");
        return;
    }
    armors_msg.armors.clear();
    marker_array_.markers.clear();
    armors_msg.header = armor_marker_.header = text_marker_.header = image_msg->header;
    armor_marker_.id = 0;
    text_marker_.id = 0;
    for (const auto & armor : armors) {
        cv::Mat rvec, tvec;
        bool success = pnp_solver_->solvePnP(armor, rvec, tvec);
        if (success) {
            // Fill basic info  
            temp_armor.type = static_cast<int>(armor.type);
            temp_armor.number = armor.number;

            // Fill pose
            temp_armor.pose.position.x = tvec.at<double>(0);
            temp_armor.pose.position.y = tvec.at<double>(1);
            temp_armor.pose.position.z = tvec.at<double>(2);
            // rvec to 3x3 rotation matrix
            cv::Mat rotation_matrix;
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

            geometry_msgs::msg::TransformStamped ts;
            ts.transform.translation.x = temp_armor.pose.position.x;
            ts.transform.translation.y = temp_armor.pose.position.y;
            ts.transform.translation.z = temp_armor.pose.position.z;
            ts.transform.rotation = temp_armor.pose.orientation;
            ts.header.stamp = image_msg->header.stamp;
            ts.header.frame_id = "camera_optical_frame";
            ts.child_frame_id = "armor";
            tf_broadcaster_->sendTransform(ts);


            // Fill the distance to image center
            temp_armor.distance_to_image_center = pnp_solver_->calculateDistanceToCenter(armor.center);

            // Fill the markers
            armor_marker_.id++;
            armor_marker_.scale.y = armor.type == ArmorType::SMALL ? 0.135 : 0.23;
            armor_marker_.pose = temp_armor.pose;
            text_marker_.id++;
            text_marker_.pose.position = temp_armor.pose.position;
            text_marker_.pose.position.y -= 0.1;
            text_marker_.text = armor.classfication_result;
            armors_msg.armors.emplace_back(temp_armor);
            marker_array_.markers.emplace_back(armor_marker_);
            marker_array_.markers.emplace_back(text_marker_);
        }
    }
    // publish
    armors_pub_->publish(armors_msg);
    // debug info
    if (params_.debug) {
        publish_debug_infos();
        text_marker_.header = armors_msg.header;
        publish_markers(armors_msg);
    }
}

void DetectorNode::energy_image_callback(sensor_msgs::msg::Image::SharedPtr image_msg) {
    ///TODO: need improve : we should restruct the energy detector and predictor with a clear line
    if (param_listener_->is_old(params_)) {
        params_ = param_listener_->get_params();
        RCLCPP_INFO(logger_, "Params updated");
    }
    if (!params_.is_armor_autoaim) {
        RCLCPP_WARN(logger_, "change state to armor!");
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/image_raw", rclcpp::SensorDataQoS(), 
            std::bind(&DetectorNode::armor_image_callback, this, std::placeholders::_1));
        return ;
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

void DetectorNode::init_markers() {
    // Visualization Marker Publisher
    // See http://wiki.ros.org/rviz/DisplayTypes/Marker
    armor_marker_.ns = "armors";
    armor_marker_.action = visualization_msgs::msg::Marker::ADD;
    armor_marker_.type = visualization_msgs::msg::Marker::CUBE;
    armor_marker_.scale.x = 0.05;
    armor_marker_.scale.z = 0.125;
    armor_marker_.color.a = 1.0;
    armor_marker_.color.g = 0.5;
    armor_marker_.color.b = 1.0;
    armor_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    text_marker_.ns = "classification";
    text_marker_.action = visualization_msgs::msg::Marker::ADD;
    text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    text_marker_.scale.z = 0.1;
    text_marker_.color.a = 1.0;
    text_marker_.color.r = 1.0;
    text_marker_.color.g = 1.0;
    text_marker_.color.b = 1.0;
    text_marker_.lifetime = rclcpp::Duration::from_seconds(0.1);

    marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/detector/markers", 10);
}

void DetectorNode::publish_markers(const autoaim_interfaces::msg::Armors& armors_msgs) {
    using Marker = visualization_msgs::msg::Marker;
    armor_marker_.action = armors_msgs.armors.empty() ? Marker::DELETE : Marker::ADD;
    marker_array_.markers.emplace_back(armor_marker_);
    marker_pub_->publish(marker_array_);
}

void DetectorNode::publish_debug_infos() {
    ///TODO: publish debug infos
    if (params_.is_armor_autoaim) {
        auto debug_images = armor_detector_->get_debug_images();
        auto result_img = debug_images.at("result_img");
        auto binary_img = debug_images.at("binary_img");
        result_img_pub_.publish(cv_bridge::CvImage(armor_marker_.header, sensor_msgs::image_encodings::RGB8, *result_img).toImageMsg()); 
        binary_img_pub_.publish(cv_bridge::CvImage(armor_marker_.header, sensor_msgs::image_encodings::MONO8, *binary_img).toImageMsg());
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

} // namespace helios_cv

// register node to component
#include "rclcpp_components/register_node_macro.hpp"

RCLCPP_COMPONENTS_REGISTER_NODE(helios_cv::DetectorNode);