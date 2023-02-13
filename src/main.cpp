#include "opencv2/calib3d.hpp"
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/opencv.hpp"
#include "opencv2/videoio.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#define DATA_PATH "data/"
#define IMAGE_PATH_DIR DATA_PATH "small-Voc2007/"

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

void postProcessing(vector<Mat> outs, Net net, Mat img) {
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
    rectangle(img, box, Scalar(0, 255, 0), 2);
    // draw prediction
  }
}

int main() {
  vector<Mat> imageMat = readImageVector(getAllImageFiles(IMAGE_PATH_DIR));
  Mat blob;

  string file = "yolo/classes_names_yolo.txt";
  ifstream ifs(file.c_str());
  if (!ifs.is_open())
    cerr << file << " not found !" << endl;

  string cfg_file   = "yolo/yolov4-tiny.cfg";
  string model_file = "yolo/yolov4-tiny.weights";

  vector<string> classes;
  string line;
  while (getline(ifs, line))
    classes.push_back(line);

  Net net = readNet(model_file, cfg_file);

  for (const auto &img : imageMat) {

    blobFromImage(img, blob, 1., Size(416, 416), Scalar(), true);
    net.setInput(blob, "", 0.00392, Scalar(0, 0, 0));

    vector<String> outNames = net.getUnconnectedOutLayersNames();
    vector<Mat> outs;

    net.forward(outs, outNames);
    postProcessing(outs, net, img);
    imshow("image", img);
    waitKey(1000);
  }
}