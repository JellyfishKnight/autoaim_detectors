#include "NetDetectors.hpp"
#include "BaseNetDetector.hpp"

namespace helios_cv {

OVNetDetector::OVNetDetector(const BaseNetDetectorParams& params) {
    params_ = params;
}

OVNetDetector::~OVNetDetector() {
    // delete infer_;
    clear_queue(futs_);
    for (int i = 0; i < params_.net_params.POOL_NUM; i++) {
        delete detect_vector_[i];
    }
}

ArmorsStamped OVNetDetector::detect_armors(const ImageStamped& image_stamped) {
    armors_stamped_.armors.clear();
    image_stamped_ = image_stamped;
    if (image_stamped_.image.empty()) {
        return armors_stamped_;
    }
    if (futs_.size() < params_.net_params.POOL_NUM) {
        detect_vector_[futs_.size()]->img_ = image_stamped_;
        futs_.push(thread_pool_->submit(&Inference::detect, &(*detect_vector_[futs_.size()])));
    } else {
        futs_.front().wait();
        armors_stamped_ = std::move(futs_.front().get());
        futs_.pop();
        // header.stamp = this->now();
        // sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "rgb8", detect_vector[frames % POOL_NUM]->img_).toImageMsg();
        // armor_pub_->publish(*msg);
        detect_vector_[frames_ % params_.net_params.POOL_NUM]->img_ = image_stamped_;
        futs_.push(thread_pool_->submit(&Inference::detect, &(*detect_vector_[frames_++ % params_.net_params.POOL_NUM])));
        return armors_stamped_;
    }
    ArmorsStamped empty_armors;
    return empty_armors;

}

void OVNetDetector::set_params(void* params) {
    params_ = *static_cast<BaseNetDetectorParams*>(params);
}

std::map<std::string, const cv::Mat*> OVNetDetector::get_debug_images() {
    std::map<std::string, const cv::Mat*> debug_images;
    debug_images.emplace("result_img", &image_stamped_.image);
    return debug_images;
}

void OVNetDetector::clear_queue(std::queue<std::future<ArmorsStamped>>& futs) {
    std::queue<std::future<ArmorsStamped>> empty;
    std::swap(futs_, empty);
}


} // namespace helios_cv