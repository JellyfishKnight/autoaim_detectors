// created by lijunqi liuhan on 2023/9/15
// Submodule of HeliosRobotSystem
// for more see document: https://swjtuhelios.feishu.cn/docx/MfCsdfRxkoYk3oxWaazcfUpTnih?from=from_copylink
#include "NetArmorDetector.hpp"
#include <autoaim_interfaces/msg/detail/debug_armors__struct.hpp>
#include <autoaim_interfaces/msg/detail/debug_lights__struct.hpp>
#include <iterator>
#include <opencv2/core/mat.hpp>
#include <tuple>
#include <vector>

namespace helios_cv {
NetArmorDetector::NetArmorDetector(const NAParams& params) {
    params_ = params;
}

NetArmorDetector::~NetArmorDetector() {
    // delete infer_;
    clear_queue(futs);
    for (int i = 0; i < POOL_NUM; i++) {
        delete detect_vector[i];
    }
}

void NetArmorDetector::set_cam_info(sensor_msgs::msg::CameraInfo::SharedPtr cam_info) {
    cam_info_ = cam_info;
    cam_center_ = cv::Point2f(cam_info_->k[2], cam_info_->k[5]);
}

void NetArmorDetector::init() {
    model_path_ = ament_index_cpp::get_package_share_directory("armor_detectors") + 
                    "/model/2x3bone-9086.onnx";
            frames = 0;
    detect_vector.clear();
    clear_queue(futs);

    infer1_ = new Inference(model_path_, params_);
    infer2_ = new Inference(model_path_, params_);
    infer3_ = new Inference(model_path_, params_);

    detect_vector.emplace_back(infer1_);
    detect_vector.emplace_back(infer2_);
    detect_vector.emplace_back(infer3_);
}

/**
 * @brief 检测主函数
 * @return 一个Armor的vector向量
*/
std::vector<Armor> NetArmorDetector::detect(const cv::Mat& image) {
    armors_.clear();
    img_ = image.clone();
    if (img_.empty()) {
        return armors_;
    }
    if (futs.size() < POOL_NUM) {
        detect_vector[futs.size()]->img_ = img_.clone();
        futs.push(pool.submit(&Inference::detect, &(*detect_vector[futs.size()])));
    } else {
        futs.front().wait();
        std::vector<Armor> targets = std::move(futs.front().get());
        futs.pop();
        std_msgs::msg::Header header;
        // header.stamp = this->now();
        // sensor_msgs::msg::Image::SharedPtr msg = cv_bridge::CvImage(header, "rgb8", detect_vector[frames % POOL_NUM]->img_).toImageMsg();
        // armor_pub_->publish(*msg);
        
        detect_vector[frames % POOL_NUM]->img_ = img_.clone();
        futs.push(pool.submit(&Inference::detect, &(*detect_vector[frames++ % POOL_NUM])));

    }

    return armors_;
}

void NetArmorDetector::set_params(const NAParams& params) {
    params_ = params;
}

ArmorType Inference::judge_armor_type(const Object& object) {
    cv::Point2f light_center1, light_center2, armor_center;
    double light_length1, light_length2;
    light_center1 = (object.apexes[0] + object.apexes[1]) / 2;
    light_center2 = (object.apexes[0] + object.apexes[1]) / 2;
    light_length1 = cv::norm(object.apexes[0] - object.apexes[1]);
    light_length2 = cv::norm(object.apexes[0] - object.apexes[1]);
    armor_center = (light_center1 + light_center2) / 2;
    double light_length_ratio = light_length1 < light_length2 ? light_length1 / light_length2
                                                            : light_length2 / light_length1;
    // Distance between the center of 2 lights (unit : light length)
    float avg_light_length = (light_length1 + light_length2) / 2;
    float center_distance = cv::norm(light_center1 - light_center2) / avg_light_length;
    // Judge armor type
    ArmorType type;
    type = center_distance > params_.armor_params.min_large_center_distance ? ArmorType::LARGE : ArmorType::SMALL;
    return type;
}


Inference::Inference(std::string model_path, const NAParams& params) {
    /*--------------openvino各个模块进行初始化---------*/
    core_.set_property(ov::cache_dir("cache"));
    std::shared_ptr<ov::Model> model = core_.read_model(model_path);
    ov::preprocess::PrePostProcessor ppp = ov::preprocess::PrePostProcessor(model);
    ppp.input().tensor().set_element_type(ov::element::u8).set_layout("NHWC").set_color_format(ov::preprocess::ColorFormat::RGB);
    ppp.input().preprocess().convert_element_type(ov::element::f32).convert_color(ov::preprocess::ColorFormat::RGB).scale({1., 1., 1.});
    ppp.input().model().set_layout("NCHW");
    ppp.output().tensor().set_element_type(ov::element::f32);
    model = ppp.build();
    complied_model_ = core_.compile_model(model, "GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));//上车后可以改成GPU（核显）
    infer_request_ = complied_model_.create_infer_request();
    input_node_ = infer_request_.get_input_tensor();
    tensor_shape_ = input_node_.get_shape();
    input_port_ = complied_model_.input();
    params_ = params;
}

/**
 * @brief 获取某一段内最大值所在位置
*/
inline int Inference::argmax(const float *ptr, int len){
    int arg_max = 0;
    for (int i = 1; i < len; i++) {
        if (ptr[i] > ptr[arg_max]) {
            arg_max = i;
        }
    }
    return arg_max;
}

float Inference::intersaction_area(const Object& a, const Object& b) {
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}



/**
 * @brief 先把图片大小进行一个预处理，变成416*416大小方便后面输入给模型
*/
cv::Mat Inference::static_resize(cv::Mat src){
    scale_ = std::min(INPUT_W/(src.cols * 1.0), INPUT_H/(src.rows * 1.0));//求出模型的输入大小（416*416）相对于原图长边的比值

    int unpad_w = scale_*src.cols;
    int unpad_h = scale_*src.rows;

    dw_ = (INPUT_W - unpad_w) / 2;
    dh_ = (INPUT_H - unpad_h) / 2;

    cv::Mat re(unpad_h, unpad_w, CV_8UC3);
    cv::resize(src, re, re.size());
    // cv::copyMakeBorder(re, re, 0, INPUT_H - unpad_h, 0, INPUT_W - unpad_w, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

    cv::Mat out;
    cv::copyMakeBorder(re, out, dh_, dh_, dw_, dw_, cv::BORDER_CONSTANT);
    return out;

}

/**
 * @brief 纯暴力循环获得各种特征图上各种步长的排列组合，一共3549种
*/
void Inference::generate_grids_and_stride(const int w, const int h, const int strides[], std::vector<GridAndStride>& grid_strides){
    for(int i=0; i<3; i++){
        int num_grid_w = w/strides[i];
        int num_grid_h = h/strides[i];
        for(int g1 = 0; g1<num_grid_h; g1++){
            for(int g0=0; g0<num_grid_w; g0++){
                grid_strides.push_back((GridAndStride{g0, g1, strides[i]}));
            }
        }
    }
}

/**
 * @brief 根据之前获得的步长网格对模型输出结果进行计算并进行一个初步的过滤
*/
void Inference::generate_yolox_proposal(std::vector<GridAndStride>& grid_strides, const float* output_buffer, float prob_threshold, std::vector<Object>& objects, float scale){
    const int num_anchors = grid_strides.size();

    for(int anchor_idx = 0; anchor_idx<num_anchors; anchor_idx++){
        const int grid0 = grid_strides[anchor_idx].grid0;
        const int grid1 = grid_strides[anchor_idx].grid1;
        const int stride = grid_strides[anchor_idx].stride;
        const int class_start = 2 * NUM_APEX + 1;

        const int basic_pos = anchor_idx*(9+NUM_CLASS+NUM_COLORS);//4个点的xy坐标+1个置信度+颜色+类别

        std::vector<cv::Point2f> point;

        for (int i = 0; i < NUM_APEX; i++) {
            point.emplace_back(cv::Point2f{((output_buffer[basic_pos + 0 + i * 2] + grid0) * stride - dw_) / scale,
                                            ((output_buffer[basic_pos + 1 + i * 2] + grid1) * stride - dh_) / scale
            });
        }
        //4个关键点
        // float x1 = (output_buffer[basic_pos+0]+grid0)*stride/scale;
        // float y1 = (output_buffer[basic_pos+1]+grid1)*stride/scale;
        // float x2 = (output_buffer[basic_pos+2]+grid0)*stride/scale;
        // float y2 = (output_buffer[basic_pos+3]+grid1)*stride/scale;
        // float x3 = (output_buffer[basic_pos+4]+grid0)*stride/scale;
        // float y3 = (output_buffer[basic_pos+5]+grid1)*stride/scale;
        // float x4 = (output_buffer[basic_pos+6]+grid0)*stride/scale;
        // float y4 = (output_buffer[basic_pos+7]+grid1)*stride/scale;
        

        //置信度最大颜色
        int color_idx = argmax(output_buffer+basic_pos+class_start, NUM_COLORS);//获取最大的颜色置信度索引
        float color_conf = output_buffer[basic_pos+class_start+color_idx];//最大的颜色置信度的值

        //置信度最大的类别
        int class_idx = argmax(output_buffer+basic_pos+class_start+NUM_COLORS, NUM_CLASS);//同上，获取类别索引
        float class_conf = output_buffer[basic_pos+class_start+NUM_COLORS+class_idx];

        //获取置信度
        float box_conf = output_buffer[basic_pos+class_start-1];
        if(box_conf>prob_threshold){
            Object obj;

            obj.apexes = std::move(point);

            obj.rect = cv::boundingRect(obj.apexes);

            obj.label = class_idx;
            obj.color = color_idx*class_start;
            obj.conf=box_conf*((class_conf+color_conf)/2);//分类置信度和识别置信度的乘积才是最后真正算出来的置信度

            objects.push_back(obj);
        }
    }
}


/**
 * @brief 对置信度递归进行一个快速排序
*/
void Inference::qsort_descent_inplace(std::vector<Object>& objects){
    if(objects.empty()){
        return;
    }
    qsort_descent_inplace(objects, 0, objects.size()-1);
}

void Inference::qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right){
    int i=left;
    int j=right;

    float p = faceobjects[(left+right)/2].conf;

    while(i<=j){
        while(faceobjects[i].conf>p){
            i++;
        }

        while(faceobjects[j].conf<p){
            j--;
        }

        if (i <= j) {
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    #pragma omp parallel sections
    {
        #pragma omp section
        {
            if(left<j)qsort_descent_inplace(faceobjects, left, j);
        }
        #pragma omp section
        {
            if(i<right)qsort_descent_inplace(faceobjects, i, right);
        }
    }
}


/**
 * @brief 非极大抑制
 * @param faceobjects 检测出来的结果
 * @param picked 非极大抑制后输出的索引就存放在这里
 * @param nms_threshold 非极大抑制阈值
*/
void Inference::nms_sorted_bboxes(std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold){
    picked.clear();
    const int n = faceobjects.size();
    std::vector<float> areas(n);

    for(int i=0; i<n; i++){
        //计算每一个面积
        // std::vector<cv::Point2f> object_apex_tmp;
        // object_apex_tmp.emplace_back(faceobjects[i].p1);
        // object_apex_tmp.emplace_back(faceobjects[i].p2);
        // object_apex_tmp.emplace_back(faceobjects[i].p3);
        // object_apex_tmp.emplace_back(faceobjects[i].p4);
        // areas[i] = cv::contourArea(object_apex_tmp);
        // areas[i] = cv::contourArea(faceobjects[i].apexes);
        areas[i] = faceobjects[i].rect.area();
    }

    for(int i=0; i<n; i++){
        Object& a = faceobjects[i];
        // std::vector<cv::Point2f>apex_a(a.apexes);
        // apex_a.emplace_back(a.p1);
        // apex_a.emplace_back(a.p2);
        // apex_a.emplace_back(a.p3);
        // apex_a.emplace_back(a.p4);


        int keep = 1;

        for(int j=0; j<(int)picked.size(); j++){
            Object &b = faceobjects[picked[j]];
            
            // std::vector<cv::Point2f> apex_inter;
            // apex_b.emplace_back(b.p1);
            // apex_b.emplace_back(b.p2);
            // apex_b.emplace_back(b.p3);
            // apex_b.emplace_back(b.p4);

            // float inter_area = cv::intersectConvexConvex(a.apexes, b.apexes, apex_inter);
            float inter_area = intersaction_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float iou = inter_area/union_area;

            if(iou>nms_threshold||isnan(iou)){
                keep=0;

                if(iou>0.9 && abs(a.conf - b.conf) < 0.15&&a.label==b.label&&a.color==b.color){
                    for (int k = 0; k < NUM_APEX; k++) {
                        b.apexes.emplace_back(a.apexes[k]);
                        //areas[i] + areas[picked[j]] - inter_area
                    }
                }
            }
        }

        if(keep){
            picked.emplace_back(i);
        }
    }
}


/**
 * @brief 获取模型的输出后对结果进行解码
 * @param output_buffer 结果的首地址
 * @param object 解码后的结果保存在这里，具体看Object的定义
 * @param scale 输入图片对于原图片的缩放比例
 * 
*/
void Inference::decode(const float *output_buffer, std::vector<Object>& objects, float scale){
    std::vector<Object>proposals;
    const int strides[3]={8, 16, 32};//步长
    std::vector<GridAndStride> grid_strides;

    generate_grids_and_stride(INPUT_W, INPUT_H, strides, grid_strides);
    generate_yolox_proposal(grid_strides, output_buffer, 0.5, proposals, scale);
    qsort_descent_inplace(proposals);
    if(proposals.size()>=128){
        proposals.resize(128);
    }

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, NMS_THRESH);

    int count = picked.size();
    step_ = count;
    objects.resize(count);
    //非极大抑制后的放入object里
    for(int i=0; i<count; i++){
        objects[i] = proposals[picked[i]];
    }

    for (auto& object : objects) {
        auto N = object.apexes.size();
        if (N >= 2 * NUM_APEX) {
            cv::Point2f fin_point[NUM_APEX];

            for (int i = 0; i < N; i++) {
                fin_point[i % NUM_APEX] += object.apexes[i];
            }

            for (int i = 0; i < NUM_APEX; i++) {
                fin_point[i].x = fin_point[i].x / (N / NUM_APEX);
                fin_point[i].y = fin_point[i].y / (N / NUM_APEX);
            }
            object.apexes.clear();
            for (int i = 0; i < NUM_APEX; i++) {
                object.apexes.emplace_back(fin_point[i]);
            } 
        }
    }

}

/**
 * @brief 检测主函数
 * @return 一个Armor的vector向量
*/
std::vector<Armor> Inference::detect(){
    std::vector<Armor> armor_;
    //对图像进行处理，使其变成可以传给模型的数据类型
    // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::Mat pre_img = static_resize(img_);
    //cv::dnn::blobFromImage(pre_img, blob, 1.0, cv::Size(INPUT_W, INPUT_H), cv::Scalar(), false, false);
    float* input_data = (float *)pre_img.data;

    //把数据传给模型
    //ov::Tensor input_tensor(input_port_.get_element_type(), input_port_.get_shape(), blob.ptr(0));
    ov::Tensor input_tensor = ov::Tensor(complied_model_.input().get_element_type(), complied_model_.input().get_shape(), input_data);
    infer_request_.set_input_tensor(input_tensor);


    //执行推理
    infer_request_.infer();
    // infer_request_.start_async();
   
    //得到推理结果
    const ov::Tensor& output = infer_request_.get_output_tensor(0);
    const float* output_buffer = output.data<const float>();

    std::vector<Object> objects;
    //对推理结果进行解码
    decode(output_buffer, objects, scale_);
    
    //设置返回结果
    for(auto &object : objects){
        if (object.conf < params_.net_classifier_thresh) {
            continue;
        }
        if (object.color != params_.is_blue) {
            continue;
        }
        Armor armor_target;
        armor_target.number = NUMBER_LABEL[object.label];
        armor_target.left_light.top = std::move(object.apexes[0]);
        armor_target.left_light.bottom = std::move(object.apexes[1]);
        armor_target.right_light.bottom = std::move(object.apexes[2]);
        armor_target.right_light.top = std::move(object.apexes[3]);

        armor_target.type = judge_armor_type(object);
        
        armor_.emplace_back(armor_target);
    }
    drawresult(armor_);

    return armor_;
}


float* Inference::thread_pre(cv::Mat &src) {
    cv::Mat pre_img = static_resize(src);
    return (float *)pre_img.data;
}

const float* Inference::thread_infer(float *input_data) {
    ov::Tensor input_tensor = ov::Tensor(complied_model_.input().get_element_type(), complied_model_.input().get_shape(), input_data);
    infer_request_.set_input_tensor(input_tensor);

    infer_request_.infer();
   
    //得到推理结果
    const ov::Tensor& output = infer_request_.get_output_tensor(0);
    const float* output_buffer = output.data<const float>();
    return output_buffer;
}

std::vector<Armor> Inference::thread_decode(const float* output_buffer) {
    std::vector<Armor> armor_;
    std::vector<Object> objects;

    //对推理结果进行解码
    decode(output_buffer, objects, scale_);
    
    //设置返回结果
    for(auto &object : objects){
        if (object.conf < params_.net_classifier_thresh) {
            continue;
        }
        if (object.color != params_.is_blue) {
            continue;
        }
        Armor armor_target;
        armor_target.number = NUMBER_LABEL[object.label];
        armor_target.left_light.top = std::move(object.apexes[0]);
        armor_target.left_light.bottom = std::move(object.apexes[1]);
        armor_target.right_light.bottom = std::move(object.apexes[2]);
        armor_target.right_light.top = std::move(object.apexes[3]);

        armor_target.type = judge_armor_type(object);

        armor_.emplace_back(armor_target);
    }
    return armor_;
}


void Inference::drawresult(std::vector<Armor> armors) {
    if (armors.empty()) {
        return;
    }
    // Draw armors
    for (const auto & armor : armors) {
        if (armor.type != ArmorType::INVALID) {
            cv::line(img_, armor.left_light.top, armor.right_light.bottom, cv::Scalar(0, 255, 0), 2);
            cv::line(img_, armor.left_light.bottom, armor.right_light.top, cv::Scalar(0, 255, 0), 2);
        }
    }

    // Show numbers and confidence
    for (const auto & armor : armors) {
        if (armor.type != ArmorType::INVALID) {
            cv::putText(
            img_, armor.classfication_result, armor.left_light.top, cv::FONT_HERSHEY_SIMPLEX, 0.8,
            cv::Scalar(0, 255, 255), 2);
        }
    }    
    return;
}



std::vector<Armor> Inference::async_detect() {
    cv::cvtColor(img_, img_, cv::COLOR_BGR2RGB);

    std::future<float*> result_input_data = std::async(std::launch::async, &Inference::thread_pre, this, std::ref(img_));
    std::future<const float*> result_output_buff = std::async(std::launch::async, &Inference::thread_infer, this, result_input_data.get());
    std::future<std::vector<Armor>> result_armor = std::async(std::launch::async, &Inference::thread_decode, this, result_output_buff.get());

    drawresult(result_armor.get());

    return result_armor.get();
}
} // helios_cv