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

#define YOLO_MODEL_FILE MODEL_PATH "yolo/yolov4-tiny.weights"
#define YOLO_CFG_FILE MODEL_PATH "yolo/yolov4-tiny.cfg"
#define YOLO_CLASS_NAMES MODEL_PATH "yolo/classes_names_yolo.txt"

using namespace cv;
using namespace std;
using namespace dnn;

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

void postProcessing(const vector<Mat> &outs, const Net &net, Mat img) {
  float confidenceThreshold = 0.33;

  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;

  for (auto &out : outs) {
    Mat outBlob = Mat(out.size(), out.depth(), out.data);

    for (int j = 0; j < outBlob.rows; j++) {
      Mat scores = outBlob.row(j).colRange(5, outBlob.cols);
      Point classIdPoint;
      double confidence;
      minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
      if (confidence > confidenceThreshold) {
        auto centerX = outBlob.row(j).at<float>(0) * img.cols;
        auto centerY = outBlob.row(j).at<float>(1) * img.rows;
        auto width   = outBlob.row(j).at<float>(2) * img.cols;
        auto height  = outBlob.row(j).at<float>(3) * img.rows;
        auto left    = centerX - width / 2;
        auto top     = centerY - height / 2;

        classIds.push_back(classIdPoint.x);
        confidences.push_back(confidence);
        rectangle(img, Rect(left, top, width, height), Scalar(0, 255, 0), 2);
      }
    }
  }
}

int main() {
  vector<Mat> imageMat =
      randomizeVectorMat(readImageVector(getAllImageFiles(IMAGE_PATH_DIR)));
  Mat blob;

  for (const auto &image : imageMat) {
    imshow("image", image);
    blobFromImage(image, blob, 1., Size(416, 416), Scalar(), true);
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
    Net net   = readNet(YOLO_MODEL_FILE, YOLO_CFG_FILE);

    for (const auto &img : imageMat) {
      int padding = 50;
      Mat padded_image(img.size().height + 2 * padding,
                       img.size().width + 2 * padding, CV_8UC3,
                       cv::Scalar(0, 0, 0));

      img.copyTo(padded_image(
          cv::Rect(padding, padding, img.size().width, img.size().height)));

      blobFromImage(padded_image, blob, 1., Size(224, 224),
                    Scalar(104, 117, 123), true);
      model.setInput(blob);
      net.setInput(blob, "", 0.00392, Scalar(0, 0, 0));
      Mat prob = model.forward();

      Point classIdPoint;
      double confidence;
      minMaxLoc(prob, nullptr, &confidence, nullptr, &classIdPoint);
      int classId                 = classIdPoint.x;

      vector<string> outNames     = model.getUnconnectedOutLayersNames();
      vector<string> outNamesYolo = net.getUnconnectedOutLayersNames();
      vector<Mat> outs;
      model.forward(outs, outNames);
      net.forward(outs, outNamesYolo);

      string label = format("%s: %2.f", classes[classId].c_str(), confidence);

      postProcessing(outs, model, padded_image);

      putText(padded_image, label, Point(0, padded_image.rows - 7),
              FONT_HERSHEY_SIMPLEX, 0.8, CV_RGB(0, 255, 0), 2, LINE_AA);

      imshow("image", padded_image);
      waitKey(1000);
    }
  }