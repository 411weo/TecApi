#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <opencv2/features2d/features2d.hpp>

// 全局变量，用于存储滑动条的值
int lowH = 0, highH = 255; // HSV的Hue范围是0-180
int lowS = 0, highS = 255; // HSV的Saturation范围是0-255
int lowV = 0, highV = 255; // HSV的Value范围是0-255
using namespace cv;

// 滑动条回调函数

int main() {
    // 创建VideoCapture对象并传入摄像头的ID
    cv::VideoCapture cap(0); // 0是默认摄像头的ID

    // 检查摄像头是否成功打开
    if (!cap.isOpened()) {
        std::cerr << "无法打开摄像头" << std::endl;
        return -1;
    }

    cv::Mat frame;
    cv::namedWindow("demo", cv::WINDOW_AUTOSIZE);

    while (true) {
        // 从摄像头读取一帧
        cap >> frame;

        // 如果正确读取帧，frame将是彩色的
        if (frame.empty()) {
            std::cerr << "无法读取帧" << std::endl;
            break;
        }

        // 显示图像

        cv::Mat flipImage;
        cv::Mat Image;

        // 水平翻转图像
        // 第二个参数是翻转代码，0代表沿x轴翻转（水平翻转）
        cv::flip(frame, flipImage, 1);//翻转图像

        //cv::cvtColor(flipImage, Image, cv::COLOR_BGR2GRAY);//转化为灰度图
        /*
        使用canny算子
        */
        //cv::Canny(flipImage, Image, 50, 150, 3);//输入图像，输出图像，最低阈值，最高阈值，sobel算子的大小(卷积核)，

        //使用高斯滤波
        //cv::GaussianBlur(flipImage, Image, cv::Size(5, 5), 0);

        //改变图片亮度
        //flipImage.convertTo(Image, -1, 1, 50);//-1表示数据类型不变，1表示缩放因子，直接在图像像素点值上进行加减改变图片亮度
        
        //色彩均衡  调整三通道的值


        //特征点检测
        // cv::Ptr<Feature2D> orb = ORB::create();
        // std::vector<KeyPoint> keypoints;
        // orb->detect(flipImage, keypoints);
        // drawKeypoints(flipImage, keypoints, Image, Scalar::all(-1), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        
        //轮廓检测
        // Mat gray,binary;
        // cvtColor(flipImage, gray, COLOR_BGR2GRAY);  // 转换为灰度图像
        // threshold(gray, binary, 128, 255, THRESH_BINARY);  // 阈值处理
        // std::vector<std::vector<Point>> contours;
        // std::vector<Vec4i> hierarchy;
        // findContours(binary, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        // drawContours(gray, contours, -1, Scalar(0, 255, 0), 2, LINE_8, hierarchy);//在图像上画轮廓 gray不能是空的
        // Image = gray;


        //检测圆
        // std::vector<Vec3f> circles;
        // Mat gray,binary;
        // Image = flipImage;
        // cvtColor(flipImage, gray, COLOR_BGR2GRAY);
        // HoughCircles(gray, circles, HOUGH_GRADIENT, 1, gray.rows/8, 200, 100, 0, 0);
        // for (size_t i = 0; i < circles.size(); i++){
        //     Vec3i c = circles[i];
        //     Point center = Point(c[0], c[1]);
        //     int radius = c[2];
        //     circle(Image, center, 3, Scalar(0, 255, 0), -1, LINE_AA);
        //     circle(Image, center, radius, Scalar(0, 0, 255), 3, LINE_AA);
        // }//绘制检测到的圆形



        // 显示结果
        cv::imshow("capvideo", Image);

        // 按esc退出循环
        if (cv::waitKey(1) == 27) {
            break;
        }
    }

    // 释放VideoCapture对象
    cap.release();
    // 关闭所有OpenCV窗口
    cv::destroyAllWindows();

    return 0;
}
