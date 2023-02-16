#include "opencv2/core.hpp"
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
#define VIDEO_PATH DATA_PATH "Video2.mp4"

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

#define MAX_CHOICES 5

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

bool isImagePath(const string &path, const string &ext) {
  // here we check if the given path is a valid image path
  return !path.substr(path.length() - 4, 4).compare(ext);
}

vector<string> getAllImageFiles(const string &path) {
  // here we return all image files in a given directory
  vector<string> imagePath;
  for (const auto &path : recursiveListFiles(path)) {
    string pathToInspect = path.path().string();
    if (isImagePath(pathToInspect, ".jpg")) {
      imagePath.push_back(pathToInspect);
    }
  }
  cout << "Found " << imagePath.size() << " image files" << endl;
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

Mat setPadding(const Mat &img) {
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

void postProcessing(const vector<Mat> &outs, Mat img,
                    const vector<string> &classNames, const Scalar &color) {
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
        boxes.emplace_back(left, top, width, height);
      }
    }
  }
  float nmsThreshold = 0.5;
  vector<int> indices;
  NMSBoxes(boxes, confidences, confidenceThreshold, nmsThreshold, indices);
  for (int idx : indices) {
    Rect box = boxes[idx];
    rectangle(img, box, GREEN, 2);

    string label =
        format("%s: %2.f", classNames[classIds[idx]].c_str(), confidences[idx]);
    putText(img, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.8, color,
            2, LINE_AA);
  }
}

void drawRoi(const Mat &img, Net &model, const vector<string> &classNames,
             const Scalar &color) {
  Mat blob;

  blobFromImage(img, blob, 1., Size(416, 416), Scalar(), true);
  model.setInput(blob, "", 0.00392, Scalar(0, 0, 0));

  vector<String> outNames = model.getUnconnectedOutLayersNames();
  vector<Mat> outs;

  model.forward(outs, outNames);
  postProcessing(outs, img, classNames, color);
}

void imshowFullScreen(const Mat &img) {
  namedWindow("image", WND_PROP_FULLSCREEN);
  setWindowProperty("image", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
  imshow("image", img);
}

void computeReadAndPredictRandomImages(const string &path, Net &model,
                                       vector<string> &classNames) {
  for (const auto &img :
       randomizeVectorMat(readImageVector(getAllImageFiles(path)))) {
    auto start = high_resolution_clock::now();

    drawRoi(img, model, classNames, GREEN);

    imshowFullScreen(img);

    cout << "total execution time :" << computeDuration(start) << " ms" << endl;
    waitKey(1000);
  }
}

unsigned int askChoice() {
  cout << "what do you want to do?"
       << "\n\t-1 predict random images"
       << "\n\t-2 make predictions from folder path which contains images"
       << "\n\t-3 make predictions from camera"
       << "\n\t-4 make predictions from random video"
       << "\n\t-5 make predictions from video path"
       << "\n enter the choice number:";

  unsigned int choice;
  cin >> choice;

  cout << "you choose " << choice << endl;
  return choice;
}

bool isChoiceOk(unsigned int &choice) {
  return choice > 0 && choice <= MAX_CHOICES;
}

unsigned int computeAskingRealChoice() {
  unsigned int choice;
  while (!isChoiceOk(choice)) {
    choice = askChoice();
    if (isChoiceOk(choice)) {
      break;
    } else {
      cerr << "Invalid choice" << endl;
    }
  }
  return choice;
}

void computeVideoCapture(VideoCapture &capture, Net model,
                         const vector<string> &classNames) {
  Mat frame;
  while (true) {
    auto start = high_resolution_clock ::now();
    capture >> frame;

    resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));

    drawRoi(frame, model, classNames, GREEN);

    imshowFullScreen(frame);

    cout << "total execution time :" << computeDuration(start) << " ms" << endl;

    if (waitKey(1) == 27)
      break;
  }
}

void manageChoices(Net &model, vector<string> &classNames,
                   unsigned int &choice) {
  VideoCapture capture;
  string path;
  switch (choice) {
  case 1:
    computeReadAndPredictRandomImages(IMAGE_PATH_DIR, model, classNames);
  case 2:
    cout << "give the folder image path : ";
    cin >> path;
    computeReadAndPredictRandomImages(path, model, classNames);
  case 3:
    capture.open(0);
    computeVideoCapture(capture, model, classNames);
  case 4:
    capture = readVideo(VIDEO_PATH);
    computeVideoCapture(capture, model, classNames);
  case 5:
    cout << "give the video path : ";
    cin >> path;
    capture = readVideo(path);
    computeVideoCapture(capture, model, classNames);
  default:
    cerr << "invalid choice" << endl;
  }
}

int main() {
  vector<string> yoloClassNames   = readClassNames(YOLO_CLASS_NAMES);
  Net yoloModel                   = readNet(YOLO_MODEL_FILE, YOLO_CFG_FILE);

  vector<string> googleClassNames = readClassNames(GOOGLE_CLASS_NAMES);
  Net googleModel                 = readNet(GOOGLE_MODEL_FILE, GOOGLE_CFG_FILE);

  unsigned int choice             = computeAskingRealChoice();

  manageChoices(yoloModel, yoloClassNames, choice);
}