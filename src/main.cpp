#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

#include <filesystem>
#include <iostream>
#include <string>

#define DATA_PATH "data/"
#define IMAGE_PATH_DIR DATA_PATH "small-Voc2007/"
using namespace cv;
using namespace std;

Mat readImage(const string &path) {
  Mat img = imread(path, IMREAD_UNCHANGED);
  if (img.empty()) {
    cerr << "can't read img at " << path << endl;
    exit(1);
  }
  return img;
}

VideoCapture readVideo(const string &path) {
  VideoCapture vid;
  vid.open(path);
  if (!vid.isOpened()) {
    cerr << "can't read vid at " << path << endl;
    exit(1);
  }
  return vid;
}

std::filesystem::recursive_directory_iterator
recursiveListFiles(const string &path) {
  return std::filesystem::recursive_directory_iterator(path);
}

bool isImagePath(const string &path) {
  string extension = ".jpg";
  return !path.substr(path.length() - 4, 4).compare(extension);
}

vector<string> getAllImageFiles(const string &path) {
  vector<string> imagePath;
  for (const auto &path : recursiveListFiles(path)) {
    string pathToInspect = path.path().string();
    if (isImagePath(pathToInspect)) {
      imagePath.push_back(pathToInspect);
      cout << path << endl;
    }
  }
  return imagePath;
}

int main() {
  vector<string> imagePaths = getAllImageFiles(IMAGE_PATH_DIR);
  cout << imagePaths.size() << endl;
}