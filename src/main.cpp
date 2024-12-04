#include <opencv2/opencv.hpp>
#include "MotionDenoiser.h"
#include "VideoIO.h"
#include <time.h>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " input.yuv output.yuv" << endl;
        return -1;
    }

    string input_yuv = argv[1];
    string output_yuv = argv[2];
    
    // Read YUV file
    double fps;
    vector<Mat> frames = GetFramesFromYUV(input_yuv, fps);
    
    if (frames.empty()) {
        cerr << "Failed to read input YUV file" << endl;
        return -1;
    }
    
    cout << "Loaded " << frames.size() << " frames at " << fps << " fps" << endl;
    
    // Create and execute denoiser
    MotionDenoiser denoiser(frames);
    vector<Mat> denoised_frames = denoiser.Execute();
    
    // Save output in YUV format
    WriteFramesToYUV(denoised_frames, output_yuv);
    
    cout << "Video denoising complete. Output saved to: " << output_yuv << endl;
    
    return 0;
}