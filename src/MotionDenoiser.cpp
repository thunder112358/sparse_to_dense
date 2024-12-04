#include "MotionDenoiser.h"
#include <iostream>

using namespace std;
using namespace cv;

MotionDenoiser::MotionDenoiser(const std::vector<cv::Mat>& frames) {
    if (frames.empty()) {
        throw runtime_error("No frames provided");
    }
    
    m_frames = frames;
    m_size = m_frames[0].size();
    m_height = m_size.height;
    m_width = m_size.width;
    m_frameNum = m_frames.size();
    
    cout << "Initializing MotionDenoiser with " << m_frameNum << " frames" << endl;
    cout << "Frame size: " << m_width << "x" << m_height << endl;
    
    // Initialize output frames
    m_denoised_frames.resize(m_frameNum);
    for (int i = 0; i < m_frameNum; i++) {
        m_denoised_frames[i].create(m_size, CV_8UC3);
    }
    
    // Initialize optical flow matrices
    map_X.resize(m_frameNum - 1);
    map_Y.resize(m_frameNum - 1);
    for (int i = 0; i < m_frameNum - 1; i++) {
        map_X[i].create(m_size, CV_32F);
        map_Y[i].create(m_size, CV_32F);
    }
    
    // Initialize temporary flow matrices
    temp_map_X.resize(2 * N);
    temp_map_Y.resize(2 * N);
    for (int i = 0; i < 2 * N; i++) {
        temp_map_X[i].create(m_size, CV_32F);
        temp_map_Y[i].create(m_size, CV_32F);
        temp_map_X[i].setTo(1);
        temp_map_Y[i].setTo(1);
    }
    
    // Initialize other matrices
    m_mask.create(m_size, CV_32FC1);
    m_dst_temp = Mat::zeros(m_size, CV_32FC3);
    m_diff = Mat::zeros(m_size, CV_32FC1);
    m_temp = Mat::zeros(m_size, CV_8UC3);
    m_mapedX = Mat::zeros(m_size, CV_32FC3);
    m_mapedY = Mat::zeros(m_size, CV_32FC3);
    m_Counter_adder = Mat::ones(m_size, CV_32F);
    
    // Initialize coordinate matrices
    formatX = Mat::zeros(m_size, CV_32F);
    formatY = Mat::zeros(m_size, CV_32F);
    for (int i = 0; i < m_height; i++) {
        for (int j = 0; j < m_width; j++) {
            formatX.at<float>(i, j) = j;
            formatY.at<float>(i, j) = i;
        }
    }
    
    // Initialize optical flow visualization
    optical_flow_img.resize(2);  // [0] for color, [1] for arrows
    optical_flow_img[0].resize(m_frameNum - 1);
    optical_flow_img[1].resize(m_frameNum - 1);
    for (int i = 0; i < m_frameNum - 1; i++) {
        optical_flow_img[0][i].create(m_size, CV_8UC3);
        optical_flow_img[1][i].create(m_size, CV_8UC3);
    }
}

void MotionDenoiser::MotionEstimation() {
    cout << "Starting motion estimation..." << endl;
    
    // Create optical flow object
    Ptr<DenseOpticalFlow> flow = DISOpticalFlow::create(DISOpticalFlow::PRESET_MEDIUM);
    
    for (int i = 1; i < m_frameNum; i++) {
        Mat prev_gray, curr_gray;
        cvtColor(m_frames[i-1], prev_gray, COLOR_BGR2GRAY);
        cvtColor(m_frames[i], curr_gray, COLOR_BGR2GRAY);
        
        Mat flow_mat;
        flow->calc(prev_gray, curr_gray, flow_mat);
        
        // Split flow into X and Y components
        vector<Mat> flow_parts;
        split(flow_mat, flow_parts);
        flow_parts[0].copyTo(map_X[i-1]);
        flow_parts[1].copyTo(map_Y[i-1]);
        
        // Generate flow visualization
        Get_optical_flow_img(map_X[i-1], map_Y[i-1], 
                           optical_flow_img[0][i-1], 
                           optical_flow_img[1][i-1]);
        
        cout << "Computed flow for frames " << i-1 << " -> " << i << endl;
    }
}

void MotionDenoiser::AbsoluteMotion(int reference) {
    // Compute absolute motion for frames before reference
    for (int i = reference - N, k = 0; i < reference && k < N; i++, k++) {
        if (i >= 0) {
            temp_map_X[k] = map_X[i];
            temp_map_Y[k] = map_Y[i];
            for (int j = i + 1; j < reference; j++) {
                temp_map_X[k] += map_X[j];
                temp_map_Y[k] += map_Y[j];
            }
        }
    }
    
    // Compute absolute motion for frames after reference
    for (int i = reference + N - 1, k = 2 * N - 1; i >= reference && k >= N; i--, k--) {
        if (i < m_frames.size() - 1) {
            temp_map_X[k] = -map_X[i];
            temp_map_Y[k] = -map_Y[i];
            for (int j = i - 1; j >= reference; j--) {
                temp_map_X[k] -= map_X[j];
                temp_map_Y[k] -= map_Y[j];
            }
        }
    }
}

void MotionDenoiser::TargetFrameBuild(int reference) {
    m_frames[reference].convertTo(m_dst_temp, CV_32FC3);
    m_Counter_adder.setTo(1);
    
    // Process frames before reference
    for (int k = reference - N, m = 0; k < reference && m < N; k++, m++) {
        if (k >= 0) {
            m_mapedX = temp_map_X[m] + formatX;
            m_mapedY = temp_map_Y[m] + formatY;
            
            remap(m_frames[k], m_temp, m_mapedX, m_mapedY, INTER_LINEAR);
            
            for (int i = 0; i < m_height; i++) {
                for (int j = 0; j < m_width; j++) {
                    Vec3b ref_pixel = m_frames[reference].at<Vec3b>(i, j);
                    Vec3b temp_pixel = m_temp.at<Vec3b>(i, j);
                    
                    int R = abs(ref_pixel[0] - temp_pixel[0]);
                    int G = abs(ref_pixel[1] - temp_pixel[1]);
                    int B = abs(ref_pixel[2] - temp_pixel[2]);
                    int Y = (R + 2 * G + B) / 4;
                    
                    float weight = Y > 40 ? 0.0f : 1.0f;
                    
                    m_dst_temp.at<Vec3f>(i, j)[0] += weight * temp_pixel[0];
                    m_dst_temp.at<Vec3f>(i, j)[1] += weight * temp_pixel[1];
                    m_dst_temp.at<Vec3f>(i, j)[2] += weight * temp_pixel[2];
                    m_Counter_adder.at<float>(i, j) += weight;
                }
            }
        }
    }
    
    // Process frames after reference
    for (int k = reference + 1, m = N; k <= reference + N && m < 2 * N; k++, m++) {
        if (k < m_frameNum) {
            m_mapedX = temp_map_X[m] + formatX;
            m_mapedY = temp_map_Y[m] + formatY;
            
            remap(m_frames[k], m_temp, m_mapedX, m_mapedY, INTER_LINEAR);
            
            for (int i = 0; i < m_height; i++) {
                for (int j = 0; j < m_width; j++) {
                    Vec3b ref_pixel = m_frames[reference].at<Vec3b>(i, j);
                    Vec3b temp_pixel = m_temp.at<Vec3b>(i, j);
                    
                    int R = abs(ref_pixel[0] - temp_pixel[0]);
                    int G = abs(ref_pixel[1] - temp_pixel[1]);
                    int B = abs(ref_pixel[2] - temp_pixel[2]);
                    int Y = (R + 2 * G + B) / 4;
                    
                    float weight = Y > 40 ? 0.0f : 1.0f;
                    
                    m_dst_temp.at<Vec3f>(i, j)[0] += weight * temp_pixel[0];
                    m_dst_temp.at<Vec3f>(i, j)[1] += weight * temp_pixel[1];
                    m_dst_temp.at<Vec3f>(i, j)[2] += weight * temp_pixel[2];
                    m_Counter_adder.at<float>(i, j) += weight;
                }
            }
        }
    }
    
    // Normalize and convert back to 8-bit
    for (int i = 0; i < m_height; i++) {
        for (int j = 0; j < m_width; j++) {
            float c = m_Counter_adder.at<float>(i, j);
            if (c > 0) {
                m_dst_temp.at<Vec3f>(i, j) /= c;
            }
        }
    }
    
    m_dst_temp.convertTo(m_denoised_frames[reference], CV_8UC3);
}

void MotionDenoiser::Get_optical_flow_img(cv::Mat &motion_X, cv::Mat &motion_Y, 
                                        cv::Mat &optical_flow_img_color, 
                                        cv::Mat &optical_flow_img_arrow) {
    optical_flow_img_color = Mat::zeros(m_size, CV_8UC3);
    optical_flow_img_arrow = Mat::zeros(m_size, CV_8UC3);
    
    for (int i = 0; i < m_height; i++) {
        for (int j = 0; j < m_width; j++) {
            float fx = motion_X.at<float>(i, j);
            float fy = motion_Y.at<float>(i, j);
            
            // Compute magnitude and angle
            float magnitude = sqrt(fx*fx + fy*fy);
            float angle = atan2(fy, fx) * 180 / M_PI;
            if (angle < 0) angle += 360;
            
            // Convert to color
            int hue = angle * 2;  // OpenCV hue range is [0, 180]
            int sat = magnitude * 255 / 10;  // Scale magnitude to [0, 255]
            if (sat > 255) sat = 255;
            
            Vec3b color(hue, sat, 255);
            cvtColor(Mat(1, 1, CV_8UC3, color), Mat(1, 1, CV_8UC3, color), COLOR_HSV2BGR);
            optical_flow_img_color.at<Vec3b>(i, j) = color;
            
            // Draw arrows for visualization
            if (i % 20 == 0 && j % 20 == 0 && magnitude > 0.5) {
                Point2f start(j, i);
                Point2f end(j + fx, i + fy);
                arrowedLine(optical_flow_img_arrow, start, end, Scalar(0, 0, 255), 1, 8, 0, 0.3);
            }
        }
    }
}

std::vector<cv::Mat> MotionDenoiser::Execute() {
    clock_t start = clock();
    
    // Step 1: Estimate motion between consecutive frames
    MotionEstimation();
    
    // Step 2: Process each frame
    for (int i = 0; i < m_frameNum; i++) {
        cout << "Processing frame " << i + 1 << "/" << m_frameNum << endl;
        
        // Compute absolute motion relative to current frame
        AbsoluteMotion(i);
        
        // Build denoised frame using temporal averaging
        TargetFrameBuild(i);
        
        // Reset counter for next frame
        m_Counter_adder.setTo(1);
    }
    
    clock_t end = clock();
    double time_per_frame = double(end - start) / CLOCKS_PER_SEC / m_frameNum;
    cout << "Average processing time per frame: " << time_per_frame << " seconds" << endl;
    
    return m_denoised_frames;
}