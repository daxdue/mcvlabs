/*  For description look into the help() function. */

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>
#include <arm_neon.h>
#include <chrono>

using namespace std;
using namespace cv;

void rgb_to_gray(const uint8_t* rgb, uint8_t* gray, int num_pixels)
{
	cout << "inside function rgb_to_gray" << endl;
	auto t1 = chrono::high_resolution_clock::now();
	for(int i=0; i<num_pixels; ++i, rgb+=3) {
		int v = (77*rgb[0] + 150*rgb[1] + 29*rgb[2]);
		gray[i] = v>>8;
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << duration << " us" << endl;
}

void image_invert(const uint8_t* rgb, uint8_t* inverted, int num_pixels) {
	int j = 0;
	auto t1 = chrono::high_resolution_clock::now();
	for(int i = 0; i < num_pixels; ++i, rgb+=3) {
		uint8_t ch1 = rgb[0];
		uint8_t ch2 = rgb[1];
		uint8_t ch3 = rgb[2];

		inverted[i] = ~ch1;
		j++;
		inverted[i+8] = ~ch2;
		j++;
		inverted[i+8] = ~ch3;
		j++;
	}
	auto t2 = chrono::high_resolution_clock::now();
	auto duration = chrono::duration_cast<chrono::microseconds>(t2-t1).count();
	cout << duration << " us" << endl;
}

void image_invert_neon(const uint8_t* image, uint8_t* inverted, int num_pixels) {
  num_pixels /=8;

  uint8x8x3_t result;

  for(int i = 0; i < num_pixels; ++i, image+=8*3, inverted+=8*3) {

    uint8x8x3_t src = vld3_u8(image);

    uint8x8_t ch1 = vmvn_u8(src.val[0]);
    uint8x8_t ch2 = vmvn_u8(src.val[1]);
    uint8x8_t ch3 = vmvn_u8(src.val[2]);

		result = {ch1, ch2, ch3};
    vst3_u8(inverted, result);
  }
}

int main(int argc,char** argv)
{
	uint8_t * rgb_arr;
	uint8_t * inverted_arr_neon;

	if (argc != 2) {
		cout << "Usage: opencv_neon image_name" << endl;
		return -1;
	}

	Mat rgb_image;
	rgb_image = imread(argv[1], /*CV_LOAD_IMAGE_COLOR*/IMREAD_COLOR);
	if (!rgb_image.data) {
		cout << "Could not open the image" << endl;
		return -1;
	}
	if (rgb_image.isContinuous()) {
		rgb_arr = rgb_image.data;
	}
	else {
		cout << "data is not continuous" << endl;
		return -2;
	}

	int width = rgb_image.cols;
	int height = rgb_image.rows;
	int num_pixels = width*height;
	Mat inverted_image_neon(height, width, CV_8UC3, Scalar(0));
	inverted_arr_neon = inverted_image_neon.data;

	//auto t1_neon = chrono::high_resolution_clock::now();
  /*image_invert_neon(rgb_arr, inverted_arr_neon, num_pixels);*/
	image_invert(rgb_arr, inverted_arr_neon, num_pixels);
	//auto t2_neon = chrono::high_resolution_clock::now();

	//auto duration_neon = chrono::duration_cast<chrono::microseconds>(t2_neon-t1_neon).count();
	//cout << "image_invert_neon" << endl;
	//cout << duration_neon << " us" << endl;

	imwrite("invert_neon.png", inverted_image_neon);

    return 0;
}
