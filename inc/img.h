#ifndef __IMG_H__
#define __IMG_H__

#include <opencv2/opencv.hpp>
#include <vector>
#include <cassert>

void write_img(std::string filename, int c, int w, int h, float const *values);
void write_img(std::string filename, int c, int w, int h, char *values);

void write_img1c(std::string filename, int w, int h, float const *values);

#endif
