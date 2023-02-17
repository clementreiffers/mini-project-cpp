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
#define VIDEO_PATH DATA_PATH "Video1.mp4"

#define MODEL_PATH "models/"

#define YOLO_MODEL_FILE MODEL_PATH "yolo/yolov4-tiny.weights"
#define YOLO_CFG_FILE MODEL_PATH "yolo/yolov4-tiny.cfg"
#define YOLO_CLASS_NAMES MODEL_PATH "yolo/classes_names_yolo.txt"

#define RED CV_RGB(255, 0, 0)
#define GREEN CV_RGB(0, 255, 0)
#define BLUE CV_RGB(0, 0, 255)
#define YELLOW CV_RGB(255, 255, 0)
#define WHITE CV_RGB(255, 255, 255)
#define PINK CV_RGB(255, 0, 255)
#define ORANGE CV_RGB(255, 128, 0)
#define VIOLET CV_RGB(255, 0, 255)

#define MAX_CHOICES 6

#define IS_CAMERA true

#define SLOW_DOWN 'a'
#define SPEED_UP 'A'
#define QUIT 'q'
#define PAUSE 32

using namespace std;

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

string setStringFormat(const string &className, double confidence) {
  return format("%s %.2f %%", className.c_str(), confidence * 100);
}

void postProcessing(const vector<Mat> &outs, Mat img,
                    const vector<string> &classNames,
                    const vector<Scalar> &colors) {
  float confidenceThreshold = 0.33;

  vector<int> classIds;
  vector<float> confidences;
  vector<Rect> boxes;
  int colorIndex = 0;

  for (auto &out : outs) {
    Mat outBlob = Mat(out.size(), out.depth(), out.data);

    for (int j = 0; j < outBlob.rows; j++) {
      Mat scores = outBlob.row(j).colRange(5, outBlob.cols);
      Point classIdPoint;
      double confidence;
      minMaxLoc(scores, nullptr, &confidence, nullptr, &classIdPoint);
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
    Rect box     = boxes[idx];
    Scalar color = colors[colorIndex];
    rectangle(img, box, color, 2);

    string label = setStringFormat(classNames[classIds[idx]], confidences[idx]);
    putText(img, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.8, color,
            2, LINE_AA);
    colorIndex++;
  }
}

void drawRoi(const Mat &img, Net &model, const vector<string> &classNames,
             const vector<Scalar> colors) {
  Mat blob;

  blobFromImage(img, blob, 1., Size(416, 416), Scalar(), true);
  model.setInput(blob, "", 0.00392, Scalar(0, 0, 0));

  vector<String> outNames = model.getUnconnectedOutLayersNames();
  vector<Mat> outs;

  model.forward(outs, outNames);
  postProcessing(outs, img, classNames, colors);
}

void manageKeys(int &key, int &speed, const Mat &img) {
  string text;
  Scalar color;
  if (key == QUIT || key == 'Q' || key == 27) {
    cout << "Exiting.." << endl;
    exit(0);
  }
  if (key == SPEED_UP) {
    if (speed - 100 > 0) {
      speed -= 100;
    } else {
      speed = 1;
    }
    text  = "Speed up: " + to_string(speed);
    color = RED;
  }
  if (key == SLOW_DOWN) {
    speed += 100;
    text  = "Slow down: " + to_string(speed);
    color = BLUE;
  }
  if (key == PAUSE) {
    if (speed > 0) {
      speed = 0;
      text  = "PAUSE ||";
      color = BLUE;
    } else if (speed == 0) {
      speed = 1000;
      text  = "LECTURE |>";
      color = RED;
    }
  }
  putText(img, text, Point(30, 30), FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
          LINE_AA);
}

void computeReadAndPredictRandomImages(const string &path, Net &model,
                                       const vector<string> &classNames,
                                       const vector<Scalar> &colors) {
  int speed = 1000;
  for (const auto &img :
       randomizeVectorMat(readImageVector(getAllImageFiles(path)))) {
    auto start = high_resolution_clock::now();

    drawRoi(img, model, classNames, colors);

    cout << "total execution time :" << computeDuration(start) << " ms" << endl;
    int key = waitKey(speed);
    manageKeys(key, speed, img);

    imshow("image", img);
  }
}

unsigned int askChoice() {
  cout << "what do you want to do?"
       << "\n\t-1 predict random images"
       << "\n\t-2 make predictions from folder path which contains images"
       << "\n\t-3 make predictions from camera"
       << "\n\t-4 make predictions from random video"
       << "\n\t-5 make predictions from video path"
       << "\n\t-6 Stop the code"
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

[[noreturn]] void computeVideoCapture(VideoCapture &capture, Net model,
                                      const vector<string> &classNames,
                                      const vector<Scalar> &colors,
                                      bool isCamera = false) {
  Mat frame;
  int speed = 1;
  while (true) {
    auto start = high_resolution_clock ::now();
    capture >> frame;

    if (isCamera) {
      resize(frame, frame, Size(1920, 1080));
    } else {
      resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
    }
    drawRoi(frame, model, classNames, colors);

    if (isCamera) {
      namedWindow("image", WND_PROP_FULLSCREEN);
      setWindowProperty("image", WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
    } else {
      namedWindow("image", WND_PROP_FULLSCREEN);
      setWindowProperty("image", WND_PROP_FULLSCREEN, WINDOW_NORMAL);
    }

    cout << "total execution time :" << computeDuration(start) << " ms" << endl;
    int key = waitKey(speed);
    manageKeys(key, speed, frame);
    imshow("image", frame);
  }
}

void manageChoices(Net &model, vector<string> &classNames, unsigned int &choice,
                   const vector<Scalar> &colors) {
  VideoCapture capture;
  string path;
  switch (choice) {
  case 1:
    computeReadAndPredictRandomImages(IMAGE_PATH_DIR, model, classNames,
                                      colors);
    break;
  case 2:
    cout << "give the folder image path : ";
    cin >> path;
    computeReadAndPredictRandomImages(path, model, classNames, colors);
    break;
  case 3:
    capture.open(0);
    computeVideoCapture(capture, model, classNames, colors, IS_CAMERA);
    break;
  case 4:
    capture = readVideo(VIDEO_PATH);
    computeVideoCapture(capture, model, classNames, colors);
    break;
  case 5:
    cout << "give the video path : ";
    cin >> path;
    capture = readVideo(path);
    computeVideoCapture(capture, model, classNames, colors);
    break;
  case 6:
    cout << "Stopping the process...";
    break;
  default:
    cerr << "invalid choice" << endl;
  }
}

int main() {
  vector<string> yoloClassNames = readClassNames(YOLO_CLASS_NAMES);
  Net yoloModel                 = readNet(YOLO_MODEL_FILE, YOLO_CFG_FILE);

  unsigned int choice           = computeAskingRealChoice();
  vector<Scalar> colors{RED, GREEN, BLUE, YELLOW, PINK, ORANGE, VIOLET, WHITE};

  manageChoices(yoloModel, yoloClassNames, choice, colors);
}