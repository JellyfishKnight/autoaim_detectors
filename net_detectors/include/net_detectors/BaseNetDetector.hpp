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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>

#include <rclcpp/time.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

// custum utilities
#include "autoaim_utilities/ThreadPool.hpp"
#include "autoaim_utilities/Armor.hpp"

#ifdef __x86_64__
// openvino
#include <openvino/openvino.hpp>
#include <ie/inference_engine.hpp>
#include <openvino/core/core.hpp>
#include <openvino/runtime/infer_request.hpp>
#elif


#endif

namespace helios_cv {

const std::vector<std::string> ARMOR_NUMBER_LABEL {
    "guard",
    "1",
    "2",
    "3",
    "4",
    "5",
    "outpost",
    "base",  // base low
    "base"   // base high
};

const std::vector<std::string> ENERGY_NUMBER_LABEL {
    ///TODO: add energy number label
    "energy_target",
    "energy_fan",
    "energy_r"
};

typedef struct BaseNetDetectorParams {
    bool is_blue;
    bool autoaim_mode;
    bool debug;
    double classifier_threshold;
    double min_large_center_distance;
    typedef struct NetParams {
        NetParams(
            const std::string& model_path,
            int num_class,
            int num_colors,
            float nms_thresh,
            int num_apex,
            int pool_num
        ) : MODEL_PATH(model_path),
            NUM_CLASS(num_class),
            NUM_COLORS(num_colors),
            NMS_THRESH(nms_thresh),
            NUM_APEX(num_apex),
            POOL_NUM(pool_num) {}
        NetParams() = default;
        std::string MODEL_PATH;
        int NUM_CLASS;//类别总数 9
        int NUM_COLORS;//颜色 2
        float NMS_THRESH;//NMS阈值 0.2
        int NUM_APEX; // 4 
        int POOL_NUM; // Thread number 3
    }NetParams; 
    NetParams net_params;
}BaseNetDetectorParams;

typedef struct ArmorsStamped {
    rclcpp::Time stamp;
    std::vector<Armor> armors;
}ArmorsStamped;

typedef struct ImageStamped {
    rclcpp::Time stamp;
    cv::Mat image;
}ImageStamped;

#ifdef __x86_64__
    class Inference{
    public:
        explicit Inference(std::string model_path, const BaseNetDetectorParams& params);
        ArmorsStamped detect();
        ArmorsStamped async_detect();

        ImageStamped img_;
    private:
        struct Object{//存储检测结果
            int label;//分类
            int color;//颜色（这俩都同上）
            float conf;//置信度
            // cv::Point2f p1, p2, p3, p4;//左上角开始逆时针四个点]
            std::vector<cv::Point2f> apexes;
            cv::Rect_<float> rect;//外接矩形，nms非极大抑制用
        };

        struct GridAndStride{//特征图大小和相对于输入图像的步长大小
            int grid0;
            int grid1;
            int stride;
        };

    private:
        int INPUT_W;
        int INPUT_H;
        BaseNetDetectorParams params_;
        float* thread_pre(cv::Mat &src);

        const float* thread_infer(float *input_data);

        ArmorsStamped thread_decode(const float* output);

        int argmax(const float* ptr, int len);
        cv::Mat static_resize(cv::Mat src);
        
        void generate_grids_and_stride(const int w, const int h, const int strides[], std::vector<GridAndStride> &grid_strides);
        void generate_yolox_proposal(std::vector<GridAndStride> &grid_strides, const float * output_buffer, float prob_threshold, std::vector<Object>& object, float scale);
        void qsort_descent_inplace(std::vector<Object> & faceobjects, int left, int right);
        void qsort_descent_inplace(std::vector<Object>& objects);
        void nms_sorted_bboxes(std::vector<Object> & faceobjects, std::vector<int>& picked, float nms_threshold);
        void decode(const float* output_buffer, std::vector<Object>& object, float scale);

        void drawresult(ArmorsStamped result);

        float intersaction_area(const Object& a, const Object& b);
        ArmorType judge_armor_type(const Object& object);
    private:
        /*----以下都是openvino的核心组件----*/
        ov::Core core_;
        ov::CompiledModel complied_model_;
        ov::InferRequest infer_request_;
        ov::Tensor input_node_;
        ov::Shape tensor_shape_;
        ov::Output<const ov::Node> input_port_;


        float scale_;//输入大小（416*416）和原图长边的比例
        int step_;

        int dw_, dh_;
    };
#elif

#endif


class BaseNetDetector {
public:
    virtual ArmorsStamped detect_armors(const ImageStamped& image_stamped) = 0;

    virtual void set_params(void* params) = 0;

    virtual void set_cam_info(const sensor_msgs::msg::CameraInfo& cam_info) {
        cam_info_ = cam_info;
        cam_center_ = cv::Point2f(cam_info_.width / 2.0, cam_info_.height / 2.0);
    };

    virtual std::map<std::string, const cv::Mat*> get_debug_images() = 0;

protected:
    // Make constructor protected to avoid create a instance of BaseNetDetector
    BaseNetDetector() = default;

    ~BaseNetDetector() = default;


    sensor_msgs::msg::CameraInfo cam_info_;
    cv::Point2f cam_center_;

    std::shared_ptr<ThreadPool> thread_pool_;
};


} // namespace helios_cv