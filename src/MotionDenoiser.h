#ifndef __MotionDenoiser__
#define __MotionDenoiser__

#include <opencv2/opencv.hpp>
#include <vector>
#include "time.h"

#define N 4  // Temporal window size for denoising
#define COLOR 1
#define ARROW 1

class MotionDenoiser {
private:
    int m_height;
    int m_width;
    int m_frameNum;
    cv::Size m_size;
    
    std::vector<cv::Mat> m_frames;  // Input frames
    std::vector<cv::Mat> m_denoised_frames;  // Output frames
    std::vector<cv::Mat> map_X, map_Y;  // Optical flow maps
    std::vector<cv::Mat> temp_map_X, temp_map_Y;  // Temporary flow maps
    std::vector<std::vector<cv::Mat>> optical_flow_img;  // Flow visualization

    cv::Mat m_mask;
    cv::Mat m_dst_temp;
    cv::Mat m_diff;
    cv::Mat m_temp;
    cv::Mat m_mapedX, m_mapedY;
    cv::Mat m_Counter_adder;
    cv::Mat formatX, formatY;

private:
    void MotionEstimation();
    void AbsoluteMotion(int reference);
    void TargetFrameBuild(int reference);
    void Get_optical_flow_img(cv::Mat &motion_X, cv::Mat &motion_Y, 
                            cv::Mat &optical_flow_img_color, 
                            cv::Mat &optical_flow_img_arrow);

public:
    MotionDenoiser(const std::vector<cv::Mat>& frames);
    std::vector<cv::Mat> Execute();
    const std::vector<std::vector<cv::Mat>>& GetOpticalFlowImages() const { 
        return optical_flow_img; 
    }
};

#endif