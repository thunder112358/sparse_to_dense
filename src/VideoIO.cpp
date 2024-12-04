#include "VideoIO.h"

vector<cv::Mat> GetFramesFromYUV(const string& filename, double &fps) {
	printf("Reading YUV file: %s\n", filename.c_str());
	vector<cv::Mat> frames;
	ifstream file(filename, ios::binary);
	
	if (!file.is_open()) {
		cerr << "Could not open YUV file: " << filename << endl;
		return frames;
	}

	// Set default fps for YUV (can be passed as parameter if needed)
	fps = 30.0;

	// Buffer for one frame
	unsigned char* yuv_buffer = new unsigned char[FRAME_SIZE];

	// Read frames until end of file
	while (file.read(reinterpret_cast<char*>(yuv_buffer), FRAME_SIZE)) {
		// Convert YUV to BGR
		cv::Mat bgr_frame = yuv420p_to_bgr(yuv_buffer);
		frames.push_back(bgr_frame.clone());
		
		// Print first few bytes for debugging
		if (frames.size() == 1) {
			printf("First frame Y plane first 10 bytes: ");
			for (int i = 0; i < 10; i++) {
				printf("%02x ", yuv_buffer[i]);
			}
			printf("\n");
		}
	}

	delete[] yuv_buffer;
	file.close();
	printf("YUV file read complete. Total frames: %zu\n", frames.size());
	return frames;
}

void WriteFramesToYUV(const vector<cv::Mat>& frames, const string& filename) {
	ofstream file(filename, ios::binary);
	if (!file.is_open()) {
		cerr << "Could not create output YUV file: " << filename << endl;
		return;
	}

	unsigned char* yuv_buffer = new unsigned char[FRAME_SIZE];

	for (size_t i = 0; i < frames.size(); i++) {
		// Convert BGR to YUV420p
		bgr_to_yuv420p(frames[i], yuv_buffer);
		
		// Write YUV data
		file.write(reinterpret_cast<char*>(yuv_buffer), FRAME_SIZE);
		
		// Print progress
		if (i % 10 == 0) {
			printf("Written frame %zu/%zu\n", i, frames.size());
		}
	}

	delete[] yuv_buffer;
	file.close();
	printf("All frames written to YUV file\n");
}

cv::Mat yuv420p_to_bgr(unsigned char* yuv_buffer) {
	// Create Y plane (full resolution)
	cv::Mat y_plane(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, yuv_buffer);
	
	// Create U plane (quarter resolution)
	cv::Mat u_plane(FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_8UC1, 
				   yuv_buffer + FRAME_WIDTH*FRAME_HEIGHT);
	
	// Create V plane (quarter resolution)
	cv::Mat v_plane(FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_8UC1, 
				   yuv_buffer + FRAME_WIDTH*FRAME_HEIGHT + (FRAME_WIDTH*FRAME_HEIGHT)/4);

	// Resize U and V planes to full resolution
	cv::Mat u_resized, v_resized;
	cv::resize(u_plane, u_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));
	cv::resize(v_plane, v_resized, cv::Size(FRAME_WIDTH, FRAME_HEIGHT));

	// Merge YUV planes
	vector<cv::Mat> yuv_planes = {y_plane, u_resized, v_resized};
	cv::Mat yuv;
	cv::merge(yuv_planes, yuv);

	// Convert to BGR
	cv::Mat bgr_frame;
	cv::cvtColor(yuv, bgr_frame, cv::COLOR_YUV2BGR);
	return bgr_frame;
}

void bgr_to_yuv420p(const cv::Mat& bgr_frame, unsigned char* yuv_buffer) {
	// Convert BGR to YUV
	cv::Mat yuv;
	cv::cvtColor(bgr_frame, yuv, cv::COLOR_BGR2YUV);

	// Split into planes
	vector<cv::Mat> yuv_planes;
	cv::split(yuv, yuv_planes);

	// Copy Y plane (full resolution)
	yuv_planes[0].copyTo(cv::Mat(FRAME_HEIGHT, FRAME_WIDTH, CV_8UC1, yuv_buffer));

	// Downsample and copy U plane
	cv::Mat u_small;
	cv::resize(yuv_planes[1], u_small, cv::Size(FRAME_WIDTH/2, FRAME_HEIGHT/2));
	u_small.copyTo(cv::Mat(FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_8UC1, 
						  yuv_buffer + FRAME_WIDTH*FRAME_HEIGHT));

	// Downsample and copy V plane
	cv::Mat v_small;
	cv::resize(yuv_planes[2], v_small, cv::Size(FRAME_WIDTH/2, FRAME_HEIGHT/2));
	v_small.copyTo(cv::Mat(FRAME_HEIGHT/2, FRAME_WIDTH/2, CV_8UC1, 
						  yuv_buffer + FRAME_WIDTH*FRAME_HEIGHT + (FRAME_WIDTH*FRAME_HEIGHT)/4));
}
