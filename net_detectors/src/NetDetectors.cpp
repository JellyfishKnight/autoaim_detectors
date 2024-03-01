#include "NetDetectors.hpp"
#include "BaseNetDetector.hpp"
#include <autoaim_utilities/ThreadPool.hpp>
#include <memory>
#include <rclcpp/logging.hpp>

namespace helios_cv {

OVNetDetector::OVNetDetector(const BaseNetDetectorParams& params) {
    params_ = params;

    thread_pool_ = std::make_shared<ThreadPool>(params_.net_params.POOL_NUM);

    frames_ = 0;
    detect_vector_.clear();
    clear_queue(futs_);

    infer1_ = new Inference(params_.net_params.MODEL_PATH, params_);
    infer2_ = new Inference(params_.net_params.MODEL_PATH, params_);
    infer3_ = new Inference(params_.net_params.MODEL_PATH, params_);

    detect_vector_.emplace_back(infer1_);
    detect_vector_.emplace_back(infer2_);
    detect_vector_.emplace_back(infer3_);
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
    armors_stamped_.stamp = image_stamped.stamp;
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
        detect_vector_[frames_ % params_.net_params.POOL_NUM]->img_ = image_stamped_;
        futs_.push(thread_pool_->submit(&Inference::detect, &(*detect_vector_[frames_++ % params_.net_params.POOL_NUM])));
        return armors_stamped_;
    }
    ArmorsStamped empty_armors;
    return empty_armors;

}

void OVNetDetector::set_params(void* params) {
    params_ = *static_cast<BaseNetDetectorParams*>(params);
    thread_pool_ = std::make_shared<ThreadPool>(params_.net_params.POOL_NUM);
    frames_ = 0;
    detect_vector_.clear();
    clear_queue(futs_);

    infer1_ = new Inference(params_.net_params.MODEL_PATH, params_);
    infer2_ = new Inference(params_.net_params.MODEL_PATH, params_);
    infer3_ = new Inference(params_.net_params.MODEL_PATH, params_);

    detect_vector_.emplace_back(infer1_);
    detect_vector_.emplace_back(infer2_);
    detect_vector_.emplace_back(infer3_);
}

std::map<std::string, const cv::Mat*> OVNetDetector::get_debug_images() {
    std::map<std::string, const cv::Mat*> debug_images;
    debug_images.emplace("result_img", &detect_vector_[frames_ % params_.net_params.POOL_NUM]->img_.image);
    return debug_images;
}

void OVNetDetector::clear_queue(std::queue<std::future<ArmorsStamped>>& futs) {
    std::queue<std::future<ArmorsStamped>> empty;
    std::swap(futs_, empty);
}


} // namespace helios_cv