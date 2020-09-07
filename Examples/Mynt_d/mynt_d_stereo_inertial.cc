
#include<iostream>
#include<algorithm>
#include<fstream>
#include<iomanip>
#include<chrono>
#include <ctime>
#include <sstream>
#include <atomic>
#include <opencv2/highgui/highgui.hpp>
#include <System.h>

#include "mynteyed/util/times.h"
#include "mynteyed/camera.h"
#include "mynteyed/utils.h"
#include "mynteyed/util/rate.h"
#include "mynteyed/device/image.h"
#include "./algorithm/counter.h"
#include "./algorithm/cam_utils.h"
#include "./algorithm/cv_painter.h"

MYNTEYE_USE_NAMESPACE;

using namespace std;

double get_machine_timestamp_s() {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::microseconds> tp = std::chrono::time_point_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now());
    auto tmp = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch());
    std::time_t timestamp = tmp.count();
//    std::cout.setf(std::ios::fixed,std::ios::floatfield);
//    std::cout<<"now_time:"<<time_stamp<<"\n";
    return (static_cast<double>(timestamp));
}

class Mynt_d_farme{
public:
    Mynt_d_farme() {
    };
    ~Mynt_d_farme() = default;

public:
    bool is_left_save,is_right_save;
    double imLeft_t,imRight_t;
    cv::Mat imLeft, imRight;
    std::vector<ORB_SLAM3::IMU::Point > imu;
};


class Slam_mynt{
public:
    Slam_mynt(){
        mynt_data=new Mynt_d_farme();
    };

    bool init();
    bool mynt_init();
    bool Mynt_Slam3();

public:
    std::atomic<Mynt_d_farme*>  mynt_data;
    std::thread mynt_init_thread,mynt_slam_thread;
};

bool Slam_mynt::mynt_init() {
    Camera cam;
    DeviceInfo dev_info;
    if (!util::select(cam, &dev_info)) {
        return 1;
    }
    util::print_stream_infos(cam, dev_info.index);
    std::cout << "Open device: " << dev_info.index << ", "<< dev_info.name << std::endl << std::endl;
    if (!cam.IsMotionDatasSupported()) {
        std::cerr << "Error: IMU is not supported on your device." << std::endl;
        return 1;
    }

    // Callbacks
    {
        // Set motion data callback
        cam.SetMotionCallback([&](const MYNTEYE_NAMESPACE::MotionData& data) {
            Mynt_d_farme* my = mynt_data.load();
            ORB_SLAM3::IMU::Point imu = ORB_SLAM3::IMU::Point(data.imu->accel[0],data.imu->accel[1],data.imu->accel[2],
                                                              data.imu->gyro[0],data.imu->gyro[1],data.imu->gyro[2],
                                                              get_machine_timestamp_s());
            my->imu.push_back(imu);
        });
    }

    OpenParams params(dev_info.index);
    {
        params.framerate = 60;
        params.color_mode = ColorMode::COLOR_RECTIFIED;
        params.stream_mode = StreamMode::STREAM_640x480;
        params.ir_intensity = 0;
    }
    cam.EnableImageInfo(false);
    cam.EnableMotionDatas(0);
    cam.Open(params);

    bool is_left_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_LEFT_COLOR);
    bool is_right_ok = cam.IsStreamDataEnabled(ImageType::IMAGE_RIGHT_COLOR);

    if (is_left_ok) cv::namedWindow("left color");
    if (is_right_ok) cv::namedWindow("right color");


    Rate rate(params.framerate);
    util::Counter counter(params.framerate);
    CVPainter painter;

    int seq=0;
    bool  allow_count = false;
    for(;seq<10;){
        cam.WaitForStream();
        allow_count = false;
        auto left_color = cam.GetStreamData(ImageType::IMAGE_LEFT_COLOR);
        if (left_color.img) {
            allow_count = true;
            cv::Mat left = left_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
            painter.DrawSize(left, CVPainter::TOP_LEFT);
            painter.DrawStreamData(left, left_color, CVPainter::TOP_RIGHT);
            painter.DrawInformation(left, util::to_string(counter.fps()),
                                    CVPainter::BOTTOM_RIGHT);
            cv::imshow("left color", left);
        }
        auto right_color = cam.GetStreamData(ImageType::IMAGE_RIGHT_COLOR);
        if (right_color.img) {
            allow_count = true;
            cv::Mat right = right_color.img->To(ImageFormat::COLOR_BGR)->ToMat();
            cv::imshow("right color", right);
        }
        if (allow_count == true) {
            counter.Update();
        }

    }
    cam.Close();
    return 0;
}

bool Slam_mynt::Mynt_Slam3() {
    std::cout<<"ORB_Slam3"<<"\n";

    std::string file_yaml = "/home/bill/ORB_SLAM/ORB_SLAM3/Examples/Mynt_d/mynteye_d_stereo.yaml";
    std::string file_orbVoc = "/home/bill/ORB_SLAM/ORB_SLAM3/Vocabulary/ORBvoc.txt";
    // Load all sequences:
    int seq=0;
    const int num_seq = 1;
    vector< vector<string> > vstrImageLeft;
    vector< vector<string> > vstrImageRight;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAcc, vGyro;
    vector< vector<double> > vTimestampsImu;
    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAcc.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    // Read rectification parameters 读取整流参数
    cv::FileStorage fsSettings(file_yaml, cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }
    cv::Mat K_l, K_r, P_l, P_r, R_l, R_r, D_l, D_r;
    fsSettings["LEFT.K"] >> K_l;
    fsSettings["RIGHT.K"] >> K_r;

    fsSettings["LEFT.P"] >> P_l;
    fsSettings["RIGHT.P"] >> P_r;

    fsSettings["LEFT.R"] >> R_l;
    fsSettings["RIGHT.R"] >> R_r;

    fsSettings["LEFT.D"] >> D_l;
    fsSettings["RIGHT.D"] >> D_r;

    int rows_l = fsSettings["LEFT.height"];
    int cols_l = fsSettings["LEFT.width"];
    int rows_r = fsSettings["RIGHT.height"];
    int cols_r = fsSettings["RIGHT.width"];

    std::cout<<rows_l<<rows_r<<cols_l<<cols_r<<" wie  \n";
    if(K_l.empty() || K_r.empty() || P_l.empty() || P_r.empty() || R_l.empty() || R_r.empty() || D_l.empty() || D_r.empty() ||
       rows_l==0 || rows_r==0 || cols_l==0 || cols_r==0)
    {
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }

    cv::Mat M1l,M2l,M1r,M2r;
    cv::initUndistortRectifyMap(K_l,D_l,R_l,P_l.rowRange(0,3).colRange(0,3),cv::Size(cols_l,rows_l),CV_32F,M1l,M2l);
    cv::initUndistortRectifyMap(K_r,D_r,R_r,P_r.rowRange(0,3).colRange(0,3),cv::Size(cols_r,rows_r),CV_32F,M1r,M2r);



    cout << endl << "-------" << endl;
    cout.precision(17);

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM3::System SLAM(file_orbVoc,file_yaml,ORB_SLAM3::System::IMU_STEREO, true);

    cv::Mat imLeft, imRight, imLeftRect, imRightRect;
    vector<ORB_SLAM3::IMU::Point> vImuMeas;
    double tframe;

    sleep(10);
    for (;seq<=10;)
    {
        Mynt_d_farme* my=mynt_data.load();

        //如果图片信息存在
        if(my->imLeft.data && my->imRight.data){
            std::cout<<"slam img \n";
            //加载imu
            for(int i=0;i<my->imu.size();i++){
                vImuMeas.push_back(ORB_SLAM3::IMU::Point(my->imu[i].a.x,my->imu[i].a.y,my->imu[i].a.z,
                                                         my->imu[i].w.x,my->imu[i].w.y,my->imu[i].w.z,
                                                         my->imu[i].t));
                std::cout<<"slam imu \n";
            }
            cv::remap(my->imLeft,imLeftRect,M1l,M2l,cv::INTER_LINEAR);
            cv::remap(my->imRight,imRightRect,M1r,M2r,cv::INTER_LINEAR);
            tframe=my->imRight_t;
            // Pass the images to the SLAM system 将图像传递给SLAM系统
            SLAM.TrackStereo(imLeftRect,imRightRect,tframe,vImuMeas);
            my->imLeft.release();
            my->imRight.release();
            my->imu.clear();
        }
    }
    // Stop all threads
    SLAM.Shutdown();
    return false;
}

bool Slam_mynt::init() {
    mynt_init_thread=std::thread(&Slam_mynt::mynt_init,this);
    sleep(10);
    //mynt_slam_thread=std::thread(&Slam_mynt::Mynt_Slam3,this);
    return false;
}

int main( ){
    Slam_mynt b;
    b.init();
    while(true){
        sleep(1);
    }
    return 0;
}
