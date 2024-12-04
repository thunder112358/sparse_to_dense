#include <cstdlib>
#include <cassert>
#include <string>
#include <fstream>
#include <limits>
#include <cmath>
#include <iostream>
#include <cstdio>
#include <opencv2/opencv.hpp>
extern "C" {
#include "iio.h"
}

// Color wheel constants
#define MAXCOLS 60
#define RAD_TO_DEG (180.0 / M_PI)

// Function to initialize color wheel
void init_color_wheel(int **color_wheel) {
    int k = 0;
    for(int i = 0; i < MAXCOLS; i++) {
        int rb = i * 255 / MAXCOLS;
        int r = rb;
        int g = 255 - rb;
        int b = 0;
        if(i >= MAXCOLS/3) {
            k = (i - MAXCOLS/3) * 255 / (MAXCOLS/3);
            r = 255 - k;
            g = k;
            b = 0;
        }
        if(i >= 2*MAXCOLS/3) {
            k = (i - 2*MAXCOLS/3) * 255 / (MAXCOLS/3);
            r = 0;
            g = 255 - k;
            b = k;
        }
        color_wheel[i][0] = r;
        color_wheel[i][1] = g;
        color_wheel[i][2] = b;
    }
}

// Function to compute color for optical flow visualization
cv::Scalar compute_color(float u, float v) {
    float rad = sqrt(u * u + v * v);
    float angle = atan2(-v, -u) * RAD_TO_DEG;
    angle += 180.0;
    
    int hue = (int)(angle * MAXCOLS / 360.0);
    if(hue < 0) hue = 0;
    if(hue >= MAXCOLS) hue = MAXCOLS-1;
    
    int sat = (int)(rad * 255 / 4);
    if(sat > 255) sat = 255;
    
    return cv::Scalar(sat, sat, sat);
}

// Function to visualize optical flow
void visualize_flow(float *flow, int nx, int ny, cv::Mat &color_map, cv::Mat &arrow_map) {
    color_map = cv::Mat(ny, nx, CV_8UC3);
    arrow_map = cv::Mat(ny, nx, CV_8UC3, cv::Scalar(255,255,255));
    
    // Visualize with color
    for(int y = 0; y < ny; y++) {
        for(int x = 0; x < nx; x++) {
            float u = flow[y*nx + x];
            float v = flow[nx*ny + y*nx + x];
            if(!std::isnan(u) && !std::isnan(v)) {
                cv::Scalar color = compute_color(u, v);
                color_map.at<cv::Vec3b>(y,x) = cv::Vec3b(color[0], color[1], color[2]);
                
                // Draw arrow every 20 pixels
                if(x % 20 == 0 && y % 20 == 0) {
                    cv::Point2f start(x, y);
                    cv::Point2f end(x + u, y + v);
                    cv::arrowedLine(arrow_map, start, end, cv::Scalar(0,0,255), 1, 8, 0, 0.3);
                }
            }
        }
    }
}

static int sparse_optical_flow(char *input, int nx, int ny, float *out) {
    float x1, x2, y1, y2;
    std::string filename_sift_matches(input);
    std::ifstream file(input);
    std::string str;

    //Initialize all the the optical flow to NAN
    for (int j = 0; j < ny; j++){
        for (int i = 0; i < nx; i++){
            out[j*nx + i] = NAN;
            out[nx*ny + j*nx + i] = NAN;
        }
    }
    if (file){
        //Insert the sparse flow obtained from matches
        while (getline(file, str)){
            //Colum I0, Row I0, Colum I1, Row I1
            sscanf(str.c_str(), "%f %f %f %f\n",
                   &x1, &y1, &x2, &y2);
            float u = x2 - x1;
            float v = y2 - y1;
            int i = std::floor(x1);
            int j = std::floor(y1);
            out[j*nx + i] = u;
            out[nx*ny + j*nx + i] = v;
        }
        return 1;
    }else{
        std::cout << "File does not exist\n";
        std::cout << input << "\n";
        return 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        fprintf(stderr, "usage:\n\t%s sift_matches.txt colum row out.flo\n", *argv);
        fprintf(stderr, "usage:\n\t Nargs:%d\n", argc);
        return 1;
    }

    char *filename_in = argv[1];
    char *filename_out = argv[4];
    int nx = atoi(argv[2]);
    int ny = atoi(argv[3]);
    float *out = new float[2*nx*ny];
    
    // Compute sparse optical flow
    sparse_optical_flow(filename_in, nx, ny, out);
    
    // Visualize flow
    cv::Mat color_map, arrow_map;
    visualize_flow(out, nx, ny, color_map, arrow_map);
    
    // Save visualizations
    std::string base_name(filename_out);
    base_name = base_name.substr(0, base_name.find_last_of("."));
    cv::imwrite(base_name + "_color.png", color_map);
    cv::imwrite(base_name + "_arrow.png", arrow_map);
    
    // Save the optical flow
    iio_save_image_float_split(filename_out, out, nx, ny, 2);

    delete [] out;
    return 0;
}

