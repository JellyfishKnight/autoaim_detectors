#include "TraditionalEnergyDetector.hpp"
#include <autoaim_utilities/Armor.hpp>
#include <map>
#include <opencv2/core.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <rclcpp/logging.hpp>
#include <string>

namespace helios_cv {

TraditionalEnergyDetector::TraditionalEnergyDetector(const TraditionalEnergyParams& params) {
    params_ = params;
    contours.clear();
    point_left.clear();
    point_right.clear();
    pts.clear();
    pts.resize(4);
}

std::vector<Armor> TraditionalEnergyDetector::detect_armors(const cv::Mat& image) {
    detect_img = image.clone();
    cv::Mat img_roi;
    // binary threshold
    binary_img = preprocess(detect_img, params_.is_blue);
    find_target_ = false;
    // find fans
    if (find_target_flow(binary_img, contours) &&
        find_target_R(contours)) {
        find_target_ = true;
        setPoint(armor_fin, circle_center_point);
        getPts(armor_fin);        
    }
    // caculate roi area
    rectangle(image, rect_roi, cv::Scalar(255, 255, 255), 2);    
    // return energy armor
    armors_.clear();
    if (find_target_) {
        Armor armor;
        armor.left_light.top = pts[0];
        armor.right_light.top = pts[1];
        armor.right_light.bottom = pts[2];
        armor.left_light.bottom = pts[3];

        armor.center = circle_center_point;
        armor.type = ArmorType::ENERGY_TARGET;
        armor.classfication_result = "energy_target";
        armors_.emplace_back(armor);
    }
    // draw debug infos
    if (params_.debug) {
        draw_results(detect_img);
    }
    return armors_;
}

void TraditionalEnergyDetector::set_params(void* params) {
    params_ = *static_cast<TraditionalEnergyParams*>(params);
}

std::map<std::string, const cv::Mat*> TraditionalEnergyDetector::get_debug_images() {
    std::map<std::string, const cv::Mat*> debug_images;
    debug_images["binary_img"] = &binary_img;
    debug_images["detect_img"] = &detect_img;
    debug_images["prepro_img"] = &prepro_img;
    return debug_images;
}

std::tuple<autoaim_interfaces::msg::DebugArmors, 
            autoaim_interfaces::msg::DebugLights> TraditionalEnergyDetector::get_debug_infos() {
    //
    return std::make_tuple(debug_armors_, debug_lights_);
}

void TraditionalEnergyDetector::draw_results(cv::Mat& img) {
    cv::circle(img, circle_center_point, 3, cv::Scalar(255, 0, 0), 1);
    float c2c = cv::norm(circle_center_point - armor_fin.center);
    float r = cv::norm(circle_center_point - armor_fin.center);
    cv::circle(img, circle_center_point, r, cv::Scalar(255, 255, 0), 2);
    for (int i = 0; i < 4; i++){
        cv::line(img, pts[i], pts[(i+1)%4], cv::Scalar(0, 255, 0), 2, cv::LINE_8);
        float p2p = cv::norm(pts[i] - circle_center_point);
        if(p2p < c2c){
            cv::line(img, pts[i], circle_center_point, cv::Scalar(0, 255, 0), 2, cv::LINE_8);
        }
    }
}

cv::Mat TraditionalEnergyDetector::preprocess(const cv::Mat& src, bool isred) {
    cv::Mat kernel_close = getStructuringElement(cv::MORPH_RECT, cv::Size(9,9));
    cv::Mat img_gry, img_th, channels[3];
    cv::split(src, channels);
    if (params_.is_blue) {
        img_gry = params_.rgb_weight_b_1 * channels[2] + 
                params_.rgb_weight_b_2 * channels[1] - 
                params_.rgb_weight_b_3 * channels[0];
    } else {
        img_gry = params_.rgb_weight_r_1 * channels[0] + 
                params_.rgb_weight_r_2 * channels[1] - 
                params_.rgb_weight_r_2 * channels[2];
    }
    cv::threshold(img_gry, img_th, params_.binary_threshold, 255, cv::THRESH_BINARY);
    cv::morphologyEx(img_th, img_th, cv::MORPH_CLOSE ,kernel_close);//闭运算让目标装甲版能连成一个整体
    return img_th;
}

bool TraditionalEnergyDetector::find_target_flow(const cv::Mat& src, std::vector<std::vector<cv::Point>> &contours) {
    cv::Mat img_target, img_left, img_right;
    std::vector<std::vector<cv::Point>> contours_left, contours_right;
    cv::RotatedRect rota_1, rota_2, rota_fin;
    float area_left=0, area_right=0, lar, shor;
    float ratio;
    cv::Point2f point_1[4], point_trans[4];
    // 找轮廓
    cv::findContours(src, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    for (auto &i : contours) {
        if (i.size() < 15) {
            continue;
        }
        // 把面积太大或者太小的去掉，这里的参数不必卡太死
        if (cv::contourArea(i) < 4000 || contourArea(i) > 10000) {
            continue;
        }
        rota_1 = minAreaRect(i);
        lar = std::max(rota_1.size.width, rota_1.size.height);
        shor = std::min(rota_1.size.width, rota_1.size.height);
        // 在用面积筛一次
        if (lar * shor < 8000 || lar * shor > 20000) {
            continue;
        }
        // 用旋转矩形的长宽比再排除一些干扰项 1.45 3.1
        if (lar / shor < 1.5 || lar / shor > 2.5) {
            continue;
        }
        //经过上面两步后剩下来的基本上就是扇页了
        rota_1.points(point_1);
        //对剩下的可能的扇页进行透射变换
        setTransform(point_1, point_trans);
        cv::Mat m = cv::getPerspectiveTransform(point_trans, dst);
        cv::warpPerspective(src, img_target, m, cv::Size(200, 100));
        //透射变换后对半裁成两部分，根据官方的尺寸装甲板刚好是扇页的一半长
        img_left = img_target(r_left);
        img_right = img_target(r_right);
        contours_left.clear();
        contours_right.clear();
        area_left = 0;
        area_right = 0;
        //对两部分分别求白色部分的总面积
        cv::findContours(img_left, contours_left, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        cv::findContours(img_right, contours_right, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        for(auto &i: contours_left){
            area_left+=contourArea(i);
        }
        for(auto &j: contours_right){
            area_right+=contourArea(j);
        }
        ratio = MAX(area_left, area_right)/MIN(area_left, area_right);//求两部分面积的比值
        // 经实际观测，非目标装甲板的扇页的比值一般在1～2,而目标能达到4点几 3.5
        if(ratio < params_.area_ratio){
            continue;
        }
        //求出目标装甲板在哪半部分后就可以设置结果了
        if(area_left>area_right){
            //target_armor.emplace_back(point_left);
            rota_far = minAreaRect(point_left);
            armor_fin=rota_far;
            rota_close = minAreaRect(point_right);
            return true;
        }else{
            //target_armor.emplace_back(point_right);
            rota_far = minAreaRect(point_right);
            armor_fin=rota_far;
            rota_close = minAreaRect(point_left);
            return true;
        }
        //break;
    }
    return false;
}

bool TraditionalEnergyDetector::find_target_R(std::vector<std::vector<cv::Point>> &contours) {
    cv::Point2f R_, possible_r;
    cv::RotatedRect rota_1;
    float point_dis;
    float max_dis = 100;
    // 根据几何关系预设一个R可能存在的坐标
    R_ = R_possible();
    for (auto &i : contours) {
        // 排除太大的目标
        if (cv::contourArea(i) > 1000) {
            continue;
        }
        rota_1 = cv::minAreaRect(i);
        // 排除比值离谱的目标
        if (MAX(rota_1.size.width, rota_1.size.height) /
            MIN(rota_1.size.width, rota_1.size.height)>4) {
            continue;
        }
        // 距离预设点太远的也排除掉
        point_dis = cv::norm(R_ - rota_1.center);
        if (point_dis < max_dis) {
            max_dis = point_dis;
            possible_r = rota_1.center;
        }
    }
    if (possible_r.x != 0 || possible_r.y != 0) {
        circle_center_point = possible_r;
        return true;
    }else{
        return false;
    }
}

void TraditionalEnergyDetector::setTransform(cv::Point2f p[], cv::Point2f d[]) {
    float dis1, dis2;
    cv::Point2f mid_1, mid_2;
    point_left.clear();
    point_right.clear();
    dis1 = cv::norm(p[0] - p[1]);
    dis2 = cv::norm(p[1] - p[2]);
    if(dis1>dis2){
        d[0] = p[0];
        d[1] = p[1];
        d[2] = p[2];
        d[3] = p[3];
        mid_1.x = (p[0].x+p[1].x)/2;
        mid_1.y = (p[0].y+p[1].y)/2;
        mid_2.x = (p[3].x+p[2].x)/2;
        mid_2.y = (p[3].y+p[2].y)/2;
        point_left.emplace_back(p[0]);
        point_left.emplace_back(p[3]);
        point_left.emplace_back(mid_1);
        point_left.emplace_back(mid_2);

        point_right.emplace_back(mid_1);
        point_right.emplace_back(mid_2);
        point_right.emplace_back(p[1]);
        point_right.emplace_back(p[2]);
    } else {
        d[0] = p[0];
        d[1] = p[3];
        d[2] = p[2];
        d[3] = p[1];
        mid_1.x = (p[2].x+p[1].x)/2;
        mid_1.y = (p[2].y+p[1].y)/2;
        mid_2.x = (p[3].x+p[0].x)/2;
        mid_2.y = (p[3].y+p[0].y)/2;
        point_left.emplace_back(p[0]);
        point_left.emplace_back(p[1]);
        point_left.emplace_back(mid_1);
        point_left.emplace_back(mid_2);

        point_right.emplace_back(mid_1);
        point_right.emplace_back(mid_2);
        point_right.emplace_back(p[2]);
        point_right.emplace_back(p[3]);
    }
}

cv::Mat TraditionalEnergyDetector::get_Roi(const cv::Mat& src) {
    cv::Mat img_roi;
    int x_roi, y_roi;
    // 如果找到R了就以R为中心画出整个能量机关的范围
    if (find_target_ == true) {
        roi_point.x = MAX(circle_center_point.x-270, 0);
        roi_point.y = MAX(circle_center_point.y-270, 0);
        x_roi = MIN(540, 1280-circle_center_point.x+270);
        y_roi = MIN(540, 1024-circle_center_point.y+270);
        rect_roi = cv::Rect((int)roi_point.x, (int)roi_point.y, x_roi, y_roi);
    }else{
        roi_point.x=0;
        roi_point.y=0;
        rect_roi = cv::Rect(0, 0, 1250, 1000);
    }
    img_roi = src(rect_roi);
    return img_roi;
}

cv::Point2f TraditionalEnergyDetector::R_possible() {
    cv::Point2f armor_vec, target_R;
    armor_vec = rota_close.center - rota_far.center;
    // 经计算，设为2基本上预设点就在R上
    target_R = rota_far.center+2*armor_vec;
    return target_R;
}

void TraditionalEnergyDetector::getPts(cv::RotatedRect &armor_fin) {
    float radian = atan2((armor_fin.center.y - circle_center_point.y), 
                        (armor_fin.center.x - circle_center_point.x));
    // 根据装甲板到R的角度重新设置装甲板旋转矩形的角度，
    // 避免因为光照什么的因素使得抖动发生影响姿态解算
    armor_fin.angle = radian * 180/ M_PI;
    cv::Point2f rectpoint[4];
    armor_fin.points(rectpoint);
    if (sqrt(pow(rectpoint[0].x - rectpoint[1].x, 2) + pow(rectpoint[0].y - rectpoint[1].y, 2)) > 
        sqrt(pow(rectpoint[1].x - rectpoint[2].x, 2) + pow(rectpoint[1].y - rectpoint[2].y, 2))) {
        //0-1为长边
        pts[0]=rectpoint[0];
        pts[1]=rectpoint[1];
        pts[2]=rectpoint[2];
        pts[3]=rectpoint[3];
    }else{
        //1-2为长边
        pts[0]=rectpoint[1];
        pts[1]=rectpoint[2];
        pts[2]=rectpoint[3];
        pts[3]=rectpoint[0];
    }
}

void TraditionalEnergyDetector::setPoint(cv::RotatedRect &armor_fin, cv::Point2f &circle_center_point) {
    armor_fin.center.x += roi_point.x;
    armor_fin.center.y += roi_point.y;
    circle_center_point.x += roi_point.x;
    circle_center_point.y += roi_point.y;
}

} // namespace helios_cv