#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define DATA_PATH "data/"
#define IMAGE_PATH_DIR DATA_PATH "small-Voc2007/"

#define MODEL_PATH "models/"

#define GOOGLE_CFG_FILE MODEL_PATH "google/bvlc_googlenet.caffemodel"
#define GOOGLE_MODEL_FILE MODEL_PATH "google/bvlc_googlenet.prototxt"
#define GOOGLE_CLASS_NAMES MODEL_PATH "google/classes_names_googlenet.txt"

using namespace cv;
using namespace std;
using namespace dnn;
using namespace std::chrono;

Mat readImage(const string &path) {
  // here we give the matrix of the image given by its path
  Mat img = imread(path, IMREAD_UNCHANGED);
  if (img.empty()) {
    cerr << "can't read img at " << path << endl;
    exit(1);
  }
  return img;
}

vector<Mat> readImageVector(const vector<string> &imagePaths) {
  // here we load inside a vector all paths given in a vector, and we
  // return it
  vector<Mat> imageMat;
  for (const string &path : imagePaths) {
    imageMat.push_back(readImage(path));
  }
  return imageMat;
}

VideoCapture readVideo(const string &path) {
  // here we read a video file by given its path
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
  // here we give the iterator of all the files in a given directory
  return std::filesystem::recursive_directory_iterator(path);
}

bool isImagePath(const string &path) {
  // here we check if the given path is a valid image path
  return !path.substr(path.length() - 4, 4).compare(".jpg");
}

vector<string> getAllImageFiles(const string &path) {
  // here we return all image files in a given directory
  vector<string> imagePath;
  for (const auto &path : recursiveListFiles(path)) {
    string pathToInspect = path.path().string();
    if (isImagePath(pathToInspect)) {
      imagePath.push_back(pathToInspect);
    }
  }
  return imagePath;
}

vector<Mat> randomizeVectorMat(vector<Mat> vec) {
  mt19937 engine(random_device{}());
  std::shuffle(vec.begin(), vec.end(), engine);
  return vec;
}

time_point<steady_clock> getNow() { return high_resolution_clock::now(); }

long long computeDuration(time_point<steady_clock> &start) {
  return duration_cast<microseconds>(getNow() - start).count() / 1000;
}

int main() {

  vector<Mat> imageMat =
      randomizeVectorMat(readImageVector(getAllImageFiles(IMAGE_PATH_DIR)));
  Mat blob;

  ifstream ifs(GOOGLE_CLASS_NAMES);
  if (!ifs.is_open()) {
    cerr << GOOGLE_CLASS_NAMES << " not found!" << endl;
  }

  vector<string> classes;
  string line;
  while (getline(ifs, line)) {
    classes.push_back(line);
  }

  Net model = readNet(GOOGLE_MODEL_FILE, GOOGLE_CFG_FILE);

  for (const auto &img : imageMat) {
    auto start = high_resolution_clock::now();

    blobFromImage(img, blob, 1., Size(224, 224), Scalar(104, 117, 123), true);
    model.setInput(blob);

    Point classIdPoint;
    double confidence;
    minMaxLoc(model.forward(), nullptr, &confidence, nullptr, &classIdPoint);
    int classId = classIdPoint.x;

    vector<Mat> outs;
    model.forward(outs, model.getUnconnectedOutLayersNames());

    string label = format("%s: %2.f", classes[classId].c_str(), confidence);

    putText(img, label, Point(0, img.rows - 7), FONT_HERSHEY_SIMPLEX, 0.8,
            CV_RGB(0, 255, 0), 2, LINE_AA);

    imshow("image", img);

    long long countMiliseconds = computeDuration(start);

    cout << "execution time in miliseconds : " << countMiliseconds << endl;
    waitKey(1000);
  }
}