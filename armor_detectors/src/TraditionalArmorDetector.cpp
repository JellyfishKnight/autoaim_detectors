// created by liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "TraditionalArmorDetector.hpp"

namespace helios_cv {
    TraditionalArmorDetector::TraditionalArmorDetector(const TAParams& params) {
        params_ = params;
        number_classifier_ = nullptr;
        number_classifier_ = std::make_shared<NumberClassifier>(
            ament_index_cpp::get_package_share_directory("armor_detectors") + "/model/mlp.onnx", 
            ament_index_cpp::get_package_share_directory("armor_detectors") + "/model/label.txt",
            params_.number_classifier_thresh);
    }
    
    void TraditionalArmorDetector::set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) {
        cam_info_ = cam_info;
        cam_center_ = cv::Point2f(cam_info_->k[2], cam_info_->k[5]);
    }

    void TraditionalArmorDetector::init() {
        
    }

    std::vector<Armor> TraditionalArmorDetector::detect(const cv::Mat& img) {
        if (number_classifier_ == nullptr) {
            RCLCPP_WARN(logger_, "Detector not initialized");
            armors_.clear();
            return armors_;
        }
        // preprocess
        binary_img_ = preprocessImage(img);
        lights_ = findLights(img, binary_img_);
        armors_ = matchLights(lights_);
        if (!armors_.empty()) {
            number_classifier_->extractNumbers(img, armors_);
            number_classifier_->classify(armors_);
        }
        return armors_;
    }

    void TraditionalArmorDetector::draw_results(cv::Mat& img) {
        // Draw Lights
        for (const auto & light : lights_) {
            cv::circle(img, light.top, 3, cv::Scalar(255, 255, 255), 1);
            cv::circle(img, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
            auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
            cv::line(img, light.top, light.bottom, line_color, 1);
        }

        // Draw armors
        for (const auto & armor : armors_) {
            cv::line(img, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
            cv::line(img, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        }

        // Show numbers and confidence
        for (const auto & armor : armors_) {
            cv::putText(
            img, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
        }
    }

    void TraditionalArmorDetector::set_params(const TAParams& params) {
        params_ = params;
    }

    cv::Mat TraditionalArmorDetector::preprocessImage(const cv::Mat & input) {
        cv::Mat gray_img;
        cv::cvtColor(input, gray_img, cv::COLOR_RGB2GRAY);

        cv::Mat binary_img;
        cv::threshold(gray_img, binary_img, params_.binary_thresh, 255, cv::THRESH_BINARY);

        return binary_img;
    }

    std::vector<Light> TraditionalArmorDetector::findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img) {
        using std::vector;
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;
        cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        vector<Light> lights;

        for (const auto & contour : contours) {
            if (contour.size() < 5) {
                continue;
            }
            auto r_rect = cv::minAreaRect(contour);
            auto light = Light(r_rect);

            if (isLight(light)) {
                auto rect = light.boundingRect();
                if (  // Avoid assertion failed
                    0 <= rect.x && 0 <= rect.width && rect.x + rect.width <= rbg_img.cols && 0 <= rect.y &&
                    0 <= rect.height && rect.y + rect.height <= rbg_img.rows) {
                    int sum_r = 0, sum_b = 0;
                    auto roi = rbg_img(rect);
                    // Iterate through the ROI
                    for (int i = 0; i < roi.rows; i++) {
                        for (int j = 0; j < roi.cols; j++) {
                            if (cv::pointPolygonTest(contour, cv::Point2f(j + rect.x, i + rect.y), false) >= 0) {
                            // if point is inside contour
                            sum_r += roi.at<cv::Vec3b>(i, j)[0];
                            sum_b += roi.at<cv::Vec3b>(i, j)[2];
                            }
                        }
                    }
                    // Sum of red pixels > sum of blue pixels ?
                    light.color = sum_r > sum_b ? RED : BLUE;
                    lights.emplace_back(light);
                }
            }   
        }
        return lights;
    }
    
    std::vector<Armor> TraditionalArmorDetector::matchLights(const std::vector<Light> & lights) {
        std::vector<Armor> armors;

        // Loop all the pairing of lights
        for (auto light_1 = lights.begin(); light_1 != lights.end(); light_1++) {
            for (auto light_2 = light_1 + 1; light_2 != lights.end(); light_2++) {
                if (light_1->color != params_.is_blue || 
                    light_2->color != params_.is_blue) {
                    continue;
                }
                if (containLight(*light_1, *light_2, lights)) {
                    continue;
                }

                auto type = isArmor(*light_1, *light_2);
                if (type != ArmorType::INVALID) {
                    auto armor = Armor(*light_1, *light_2);
                    armor.type = type;
                    armors.emplace_back(armor);
                }
            }
        }

        return armors;
    }

    bool TraditionalArmorDetector::isLight(const Light & possible_light) {
        // The ratio of light (short side / long side)
        float ratio = possible_light.width / possible_light.length;
        bool ratio_ok = params_.light_params.min_ratio < ratio && 
                        ratio < params_.light_params.max_ratio;

        bool angle_ok = possible_light.tilt_angle < params_.light_params.max_angle;

        bool is_light = ratio_ok && angle_ok;

        return is_light;

    }
    
    bool TraditionalArmorDetector::containLight(
        const Light & light_1, const Light & light_2, const std::vector<Light> & lights) {
        auto points = std::vector<cv::Point2f>{light_1.top, light_1.bottom, light_2.top, light_2.bottom};
        auto bounding_rect = cv::boundingRect(points);

        for (const auto & test_light : lights) {
            if (test_light.center == light_1.center || test_light.center == light_2.center) {
                continue;
            }
            if (bounding_rect.contains(test_light.top) || 
                bounding_rect.contains(test_light.bottom) ||
                bounding_rect.contains(test_light.center)) {
                return true;
            }
        }

        return false;
    }
    
    ArmorType TraditionalArmorDetector::isArmor(const Light & light_1, const Light & light_2) {
        // Ratio of the length of 2 lights (short side / long side)
        float light_length_ratio = light_1.length < light_2.length ? light_1.length / light_2.length
                                                                    : light_2.length / light_1.length;
        bool light_ratio_ok = light_length_ratio > params_.armor_params.min_light_ratio;

        // Distance between the center of 2 lights (unit : light length)
        float avg_light_length = (light_1.length + light_2.length) / 2;
        float center_distance = cv::norm(light_1.center - light_2.center) / avg_light_length;
        bool center_distance_ok = (params_.armor_params.min_small_center_distance <= center_distance &&
                                    center_distance < params_.armor_params.max_small_center_distance) ||
                                    (params_.armor_params.min_large_center_distance <= center_distance &&
                                    center_distance < params_.armor_params.max_large_center_distance);
        // Angle of light center connection
        cv::Point2f diff = light_1.center - light_2.center;
        float angle = std::abs(std::atan(diff.y / diff.x)) / CV_PI * 180;
        bool angle_ok = angle < params_.armor_params.max_angle;

        bool is_armor = light_ratio_ok && center_distance_ok && angle_ok;
        if (!light_ratio_ok) {
            // RCLCPP_INFO(logger_, "light ratio fail");
            // RCLCPP_INFO(logger_, "light_length_ratio: %f", light_length_ratio);
        }
        if (!center_distance_ok) {
            // RCLCPP_INFO(logger_, "center_distance_fail");
            // RCLCPP_INFO(logger_, "center_distance: %f", center_distance);
        }
        if (!angle_ok) {
            // RCLCPP_DEBUG(logger_, "angle_fail");
            // RCLCPP_INFO(logger_, "angle: %f", angle);
        }
        // Judge armor type
        ArmorType type;
        if (is_armor) {
            // std::cout << "center_distance: " << center_distance << std::endl;
            type = center_distance > params_.armor_params.min_large_center_distance ? 
                                                                ArmorType::LARGE : ArmorType::SMALL;
        } else {
            type = ArmorType::INVALID;
        }
        return type;
    }

} // namespace helios_cv