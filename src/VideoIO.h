#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <fstream>

using namespace std;

// New YUV frame size constants
#define FRAME_WIDTH 1280
#define FRAME_HEIGHT 720
// YUV420 format: Y plane (WxH) + U plane (W/2 x H/2) + V plane (W/2 x H/2)
#define FRAME_SIZE (FRAME_WIDTH * FRAME_HEIGHT + (FRAME_WIDTH * FRAME_HEIGHT / 2))

vector<cv::Mat> GetFramesFromYUV(const string& filename, double &fps);
void WriteFramesToYUV(const vector<cv::Mat>& frames, const string& filename);

// Helper functions for YUV conversion
cv::Mat yuv420p_to_bgr(unsigned char* yuv_buffer);
void bgr_to_yuv420p(const cv::Mat& bgr_frame, unsigned char* yuv_buffer);