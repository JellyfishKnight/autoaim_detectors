#include "TraditionalArmorDetector.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <autoaim_utilities/NumberClassifier.hpp>
#include <memory>
#include <rclcpp/logging.hpp>
#include <tuple>

namespace helios_cv {

TraditionalArmorDetector::TraditionalArmorDetector(const TraditionalArmorParams& params) {
    params_ = params;
    auto model_path = ament_index_cpp::get_package_share_directory("traditional_detectors") + "/model/mlp.onnx";
    auto label_path = ament_index_cpp::get_package_share_directory("traditional_detectors") + "/model/label.txt";
    number_classifier_ = std::make_shared<NumberClassifier>(
        model_path,
        label_path,
        params_.number_classifier_thresh,
        std::vector<std::string>{"negative"}
    );
}

std::vector<Armor> TraditionalArmorDetector::detect_armors(const cv::Mat& image) {
    if (number_classifier_ == nullptr) {
        RCLCPP_WARN(logger_, "Detector not initialized");
        armors_.clear();
        return armors_;
    }
    // preprocess
    binary_img_ = preprocessImage(image);
    lights_ = findLights(image, binary_img_);
    armors_ = matchLights(lights_);
    if (!armors_.empty()) {
        number_classifier_->extractNumbers(image, armors_);
        number_classifier_->classify(armors_);
    }
    // debug infos
    if (params_.debug) {
        draw_results();
    }
    return armors_;
}


void TraditionalArmorDetector::set_params(void* params) {
    params_ = *static_cast<TraditionalArmorParams*>(params);
}

std::map<std::string, const cv::Mat*> TraditionalArmorDetector::get_debug_images() {
    std::map<std::string, const cv::Mat*> debug_images;
    debug_images.emplace("result_img", &result_img_);
    debug_images.emplace("binary_img", &binary_img_);
    debug_images.emplace("number_img", &number_img_);
    return debug_images;
}

std::tuple<autoaim_interfaces::msg::DebugArmors, 
            autoaim_interfaces::msg::DebugLights> TraditionalArmorDetector::get_debug_infos() {
    return std::make_tuple(debug_armors_, debug_lights_);
}

cv::Mat TraditionalArmorDetector::preprocessImage(const cv::Mat & input) {
    cv::Mat gray_img;
    cv::cvtColor(input, gray_img, cv::COLOR_RGB2GRAY);

    cv::Mat binary_img;
    cv::threshold(gray_img, binary_img, params_.binary_threshold, 255, cv::THRESH_BINARY);

    result_img_ = input.clone();
    return binary_img;
}

std::vector<Light> TraditionalArmorDetector::findLights(const cv::Mat & rbg_img, const cv::Mat & binary_img) {
    using std::vector;
    vector<vector<cv::Point>> contours;
    vector<cv::Vec4i> hierarchy;
    cv::findContours(binary_img, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    vector<Light> lights;
    debug_lights_.data.clear();

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

    debug_armors_.data.clear();

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

    // Fill debug lights info 
    autoaim_interfaces::msg::DebugLight debug_light;
    debug_light.center_x = possible_light.center.x;
    debug_light.ratio = ratio;
    debug_light.angle = possible_light.tilt_angle;
    debug_light.is_light = is_light;
    debug_lights_.data.emplace_back(debug_light);

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

    // Judge armor type
    ArmorType type;
    if (is_armor) {
        // std::cout << "center_distance: " << center_distance << std::endl;
        type = center_distance > params_.armor_params.min_large_center_distance ? 
                                                            ArmorType::LARGE : ArmorType::SMALL;
    } else {
        type = ArmorType::INVALID;
    }
    // if (!angle_ok && params_.debug) {
    //     RCLCPP_WARN(logger_, "now angle %f, thresh %f", angle, params_.armor_params.max_angle);
    // }
    // if (!center_distance_ok && params_.debug) {
    //     RCLCPP_WARN(logger_, "now distance %f, thresh %f", center_distance, params_.armor_params.max_large_center_distance);
    // }
    // if (!light_length_ratio && params_.debug) {
    //     RCLCPP_WARN(logger_, "now ratio %f, thresh %f", light_length_ratio, params_.armor_params.min_light_ratio);
    // }
    // Fill debug armors info
    autoaim_interfaces::msg::DebugArmor debug_armor;
    debug_armor.center_x = (light_1.center.x + light_2.center.x) / 2;
    debug_armor.light_ratio = light_length_ratio;
    debug_armor.center_distance = center_distance;
    debug_armor.angle = angle;
    debug_armor.type = ARMOR_TYPE_STR[static_cast<int>(type)];
    debug_armors_.data.emplace_back(debug_armor);

    return type;
}

void TraditionalArmorDetector::draw_results() {
    // Draw Lights
    for (const auto & light : lights_) {
        cv::circle(result_img_, light.top, 3, cv::Scalar(255, 255, 255), 1);
        cv::circle(result_img_, light.bottom, 3, cv::Scalar(255, 255, 255), 1);
        auto line_color = light.color == RED ? cv::Scalar(255, 255, 0) : cv::Scalar(255, 0, 255);
        cv::line(result_img_, light.top, light.bottom, line_color, 1);
    }

    // Draw armors
    for (const auto & armor : armors_) {
        if (armor.type != ArmorType::INVALID) {
            cv::line(result_img_, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
            cv::line(result_img_, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Show numbers and confidence
    for (const auto & armor : armors_) {
        if (armor.type != ArmorType::INVALID) {
            cv::putText(
            result_img_, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
        }
    }    
    // Draw image center
    cv::circle(result_img_, cam_center_, 5, cv::Scalar(255, 0, 0), 2);
    // Draw latency
    std::stringstream latency_ss;
    // latency_ss << "Latency: " << std::fixed << std::setprecision(2) << latency_ << "ms";
    auto latency_s = latency_ss.str();
    cv::putText(
        result_img_, latency_s, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    get_all_number_images();
}

void TraditionalArmorDetector::get_all_number_images() {
    // Get all number imgs
    std::vector<cv::Mat> all_number_imgs;
    if (armors_.empty()) {
        number_img_ = cv::Mat(cv::Size(20, 28), CV_8UC1);
    } else {
        all_number_imgs.reserve(armors_.size());
        for (auto armor : armors_) {
            all_number_imgs.emplace_back(armor.number_img);
        }
        cv::vconcat(all_number_imgs, number_img_);
    }
}


} // namespace helios_cv