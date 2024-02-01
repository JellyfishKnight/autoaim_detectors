#include "TraditionalEnergyDetector.hpp"

namespace helios_cv {

TraditionalEnergyDetector::TraditionalEnergyDetector(const TraditionalEnergyParams& params) {
    params_ = params;
}

std::vector<Armor> TraditionalEnergyDetector::detect_armors(const cv::Mat& image) {

}

void TraditionalEnergyDetector::set_params(void* params) {
    params_ = *static_cast<TraditionalEnergyParams*>(params);
}

std::map<std::string, const cv::Mat*> TraditionalEnergyDetector::get_debug_images() {

}

std::tuple<autoaim_interfaces::msg::DebugArmors, 
            autoaim_interfaces::msg::DebugLights> TraditionalEnergyDetector::get_debug_infos() {
    //
    return std::make_tuple(debug_armors_, debug_lights_);
}

void TraditionalEnergyDetector::draw_results(cv::Mat& img) {

}

cv::Mat TraditionalEnergyDetector::preprocess(const cv::Mat& src, bool isred) {

}

bool TraditionalEnergyDetector::find_target_flow(const cv::Mat& src, std::vector<std::vector<cv::Point>> &contours) {

}

bool TraditionalEnergyDetector::find_target_R(std::vector<std::vector<cv::Point>> &contours) {

}

void TraditionalEnergyDetector::setTransform(cv::Point2f p[], cv::Point2f d[]) {

}

float TraditionalEnergyDetector::distance(cv::Point2f p1, cv::Point2f p2) {

}

cv::Mat TraditionalEnergyDetector::get_Roi(const cv::Mat& src) {

}

cv::Point2f TraditionalEnergyDetector::R_possible() {

}

void TraditionalEnergyDetector::getPts(cv::RotatedRect &armor_fin) {

}

void TraditionalEnergyDetector::setPoint(cv::RotatedRect &armor_fin, cv::Point2f &circle_center_point) {
    
}

} // namespace helios_cv