#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <utility>
#include <vector>

#define DATA_PATH "data/"
#define IMAGE_PATH_DIR DATA_PATH "small-Voc2007/"

#define MODEL_PATH "models/"

#define GOOGLE_CFG_FILE MODEL_PATH "google/bvlc_googlenet.caffemodel"
#define GOOGLE_MODEL_FILE MODEL_PATH "google/bvlc_googlenet.prototxt"
#define GOOGLE_CLASS_NAMES MODEL_PATH "google/classes_names_googlenet.txt"

#define YOLO_MODEL_FILE MODEL_PATH "yolo/yolov4-tiny.weights"
#define YOLO_CFG_FILE MODEL_PATH "yolo/yolov4-tiny.cfg"
#define YOLO_CLASS_NAMES MODEL_PATH "yolo/classes_names_yolo.txt"

#define IMAGE_PADDING 20

#define RED CV_RGB(255, 0, 0)
#define GREEN CV_RGB(0, 255, 0)
#define BLUE CV_RGB(0, 0, 255)

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

void postProcessing(vector<Mat> outs, Net net, Mat img, Scalar color) {
  float confidenceThreshold = 0.33;

  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  for (int i = 0; i < outs.size(); i++) {
    Mat outBlob = Mat(outs[i].size(), outs[i].depth(), outs[i].data);

    for (int j = 0; j < outBlob.rows; j++) {
      Mat scores = outBlob.row(j).colRange(5, outBlob.cols);
      Point classIdPoint;
      double confidence;
      minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
      if (confidence > confidenceThreshold) {
        int centerX = outBlob.row(j).at<float>(0) * img.cols;
        int centerY = outBlob.row(j).at<float>(1) * img.rows;
        int width   = outBlob.row(j).at<float>(2) * img.cols;
        int height  = outBlob.row(j).at<float>(3) * img.rows;
        int left    = centerX - width / 2;
        int top     = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back(confidence);
        boxes.push_back(Rect(left, top, width, height));
      }
    }
  }

  float nmsThreshold = 0.5;
  vector<int> indices;
  NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);
  for (int i = 0; i < indices.size(); i++) {
    int idx  = indices[i];
    Rect box = boxes[idx];
    rectangle(img, box, color, 2);
    // draw prediction
  }
}

vector<string> readClassNames(const string &fileName) {
  vector<string> classNames;
  std::ifstream file(fileName);
  if (file.is_open()) {
    std::string className;
    while (getline(file, className)) {
      classNames.push_back(className);
    }
  }
  return classNames;
}

Mat setPadding(Mat img) {
  int padding = 50;
  Mat padded_image(img.size().height + 2 * padding,
                   img.size().width + 2 * padding, CV_8UC3,
                   cv::Scalar(0, 0, 0));

  img.copyTo(padded_image(
      cv::Rect(padding, padding, img.size().width, img.size().height)));
  return padded_image;
}

string setStringFormat(const string &className, double confidence) {
  return format("%s %.2f", className.c_str(), confidence);
}

void drawRoi(const Mat &img, Net &model, Scalar color) {
  Mat blob;

  blobFromImage(img, blob, 1., Size(416, 416), Scalar(), true);
  model.setInput(blob, "", 0.00392, Scalar(0, 0, 0));

  vector<String> outNames = model.getUnconnectedOutLayersNames();
  vector<Mat> outs;

  model.forward(outs, outNames);
  postProcessing(outs, model, img, std::move(color));
}

void drawPredictions(const Mat &img, Net &model, vector<string> &classNames,
                     Scalar color) {
  Mat blob;
  blobFromImage(img, blob, 1., Size(224, 224), Scalar(104, 117, 123), true);
  model.setInput(blob);
  Mat prob = model.forward();

  Point classIdPoint;
  double confidence;
  minMaxLoc(prob, nullptr, &confidence, nullptr, &classIdPoint);
  int classId             = classIdPoint.x;

  vector<string> outNames = model.getUnconnectedOutLayersNames();
  vector<Mat> outs;
  model.forward(outs, outNames);

  string label = format("%s: %2.f", classNames[classId].c_str(), confidence);

  putText(img, label, Point(0, img.rows - 7), FONT_HERSHEY_SIMPLEX, 0.8,
          std::move(color), 2, LINE_AA);
}

int main() {
  vector<Mat> imageMat = readImageVector(getAllImageFiles(IMAGE_PATH_DIR));

  vector<string> yoloClassNames   = readClassNames(YOLO_CLASS_NAMES);
  Net yoloModel                   = readNet(YOLO_MODEL_FILE, YOLO_CFG_FILE);

  vector<string> googleClassNames = readClassNames(GOOGLE_CLASS_NAMES);
  Net googleModel                 = readNet(GOOGLE_MODEL_FILE, GOOGLE_CFG_FILE);

  for (const auto &img : imageMat) {
    drawRoi(img, yoloModel, GREEN);
    drawPredictions(img, googleModel, googleClassNames, GREEN);
    imshow("image", img);
    waitKey(1000);
  }
}