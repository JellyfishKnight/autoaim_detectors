// created by lijunqi, liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#pragma once

#include <thread>
#include <future>
#include <memory>
#include <functional>
#include <string>
// ros
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
// openvino
#include <openvino/openvino.hpp>
#include <ie/inference_engine.hpp>
#include <openvino/core/core.hpp>
#include <openvino/runtime/infer_request.hpp>
// opencv
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
// tf2 
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2/convert.h>
// interfaces
#include "autoaim_interfaces/msg/armors.hpp"
#include <sensor_msgs/msg/camera_info.hpp>
#include <vector>
// utilities
#include "autoaim_utilities/Armor.hpp"
#include "autoaim_utilities/ThreadPool.hpp"
#include "BaseArmorDetector.hpp"

constexpr int INPUT_W = 416;//输入图片的宽
constexpr int INPUT_H = 416;//输入图片的高
constexpr int NUM_CLASS = 9;//类别总数
constexpr int NUM_COLORS = 2;//颜色
constexpr float NMS_THRESH = 0.2;//NMS阈值
constexpr int NUM_APEX = 4;
constexpr int POOL_NUM = 3; // Thread number

const std::vector<std::string> NUMBER_LABEL {
    "guard",
    "1",
    "2",
    "3",
    "4",
    "5",
    "output",
    "base",  // base low
    "base"   // base high
};


namespace helios_cv {

typedef struct NetArmorParams : public BaseArmorParams {
    int net_classifier_thresh;
    typedef struct ArmorParams {
        double min_light_ratio;
        double max_light_ratio;
        double min_small_center_distance;
        double max_small_center_distance;
        double min_large_center_distance;
        double max_large_center_distance;
        double max_angle;
    }ArmorParams;
    ArmorParams armor_params;

}NAParams;

class Inference{
public:
    Inference(std::string model_path, const NAParams& params);
    std::vector<Armor> detect();
    std::vector<Armor> async_detect();

    cv::Mat img_;
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
    NAParams params_;
    float* thread_pre(cv::Mat &src);

    const float* thread_infer(float *input_data);

    std::vector<Armor> thread_decode(const float* output);

    int argmax(const float* ptr, int len);
    cv::Mat static_resize(cv::Mat src);
    void generate_grids_and_stride(const int w, const int h, const int strides[], std::vector<GridAndStride> &grid_strides);
    void generate_yolox_proposal(std::vector<GridAndStride> &grid_strides, const float * output_buffer, float prob_threshold, std::vector<Object>& object, float scale);
    void qsort_descent_inplace(std::vector<Object> & faceobjects, int left, int right);
    void qsort_descent_inplace(std::vector<Object>& objects);
    void nms_sorted_bboxes(std::vector<Object> & faceobjects, std::vector<int>& picked, float nms_threshold);
    void decode(const float* output_buffer, std::vector<Object>& object, float scale);

    void drawresult(std::vector<Armor> result);

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


class NetArmorDetector : public BaseArmorDetector {
public:
    NetArmorDetector(const NAParams& params);

    ~NetArmorDetector();

    void init() override;

    std::vector<Armor> detect(const cv::Mat& image) override;

    void draw_results(cv::Mat& img);

    void set_params(const NAParams& params);

    void set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) override;

    std::tuple<const cv::Mat*, const cv::Mat*, const cv::Mat*> get_debug_images() override;

    std::tuple<const autoaim_interfaces::msg::DebugLights*,
                    const autoaim_interfaces::msg::DebugArmors*> get_debug_msgs() override;       
private:
    void clear_queue(std::queue<std::future<std::vector<Armor>>>& futs);
    std::string model_path_;

    cv::Mat img_;

    Inference* infer1_;
    Inference* infer2_;
    Inference* infer3_;

    ThreadPool pool{POOL_NUM};

    std::vector<Inference*> detect_vector;
    std::queue<std::future<std::vector<Armor>>> futs;

    int frames;

    std::vector<Armor> armors_;
    // params
    NAParams params_;
    // camera info
    cv::Point2f cam_center_;
    sensor_msgs::msg::CameraInfo::SharedPtr cam_info_;
};




} // namespace helios_cv