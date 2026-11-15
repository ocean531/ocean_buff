#include "main.hpp"


int main(int argc, char **argv)
{

    const fs::path root = projectRoot(argc, argv);

    cv::FileStorage detector_config(
            root /"configs/detector/detector.yaml",
            cv::FileStorage::READ
        );
        cv::FileStorage tracker_config(
            root /"configs/tracker/tracker.yaml",
            cv::FileStorage::READ
        );
        cv::FileStorage predictor_config(
            root /"configs/predictor/predictor.yaml",
            cv::FileStorage::READ
        );
        // if (!detector_config.isOpened() || !tracker_config.isOpened() ||
        //     !predictor_config.isOpened()) {
        //     throw std::runtime_error("failed to open config yaml in " root);
        // }
    
    const cv::Mat kCameraMatrix = (cv::Mat_<double>(3, 3) << 800.0, 0.0, 640.0,
                                                               0.0, 800.0, 384.0,
                                                               0.0, 0.0, 1.0);
    const cv::Mat kDistortion = cv::Mat::zeros(1, 5, CV_64F);

    buff_algo::BuffSelector selector(0); //扇叶选择器
    selector.set_camera_matrix(kCameraMatrix, kDistortion); //后面改成从yaml中读取

    buff_algo::BuffPnPSolver solver; //解算器
    solver.set_camera_matrix(kCameraMatrix, kDistortion);//同上,需要修改

    buff_algo::BuffTracker tracker(tracker_config["buff_tracker"]);
    tracker.set_mode(static_cast<buff_algo::BuffMode>(2)); //通信接口 2大1小符

    buff_algo::BuffPredictor predictor(predictor_config["buff_predictor"]);


    const fs::path config = argc > 2
                                ? fs::absolute(fs::path(argv[2]))
                                : root / "configs/detect_config.json";
    const fs::path output_dir = root / "result";


    //声明识别器
    const fs::path models = root / "所有模型/openvino";
    RuneDetector detector(
        JsonConfig{config.string(), "rune_detect", models.string()},0);
    
    std::uint64_t submitted = 0;
    cv::Mat frame;
    int processed = 0;
    buff_algo::StatusType last_status = buff_algo::StatusType::LOST;
    
    VideoClip clip = openMiddleClip(root / "测试视频/符1.avi");
    //auto capture = openVideo(root / "测试视频/符.avi");
    //std::optional<RuneDetectionFrame> rune = runRune(config, 0 , root , output_dir , frame);

    while(submitted < clip.frame_limit && clip.video.read(frame))
    { //主循环入口

    ++submitted;
    //capture.read(frame);
    auto rune = run(detector , frame , 1);

    cv::imshow("img" , rune->image);
    if (cv::waitKey(1) == 27)  // ESC 键退出
            break;

    
    std::vector<buff_algo::BuffDetection> detections ;//算法需要的类型
    detections.reserve(100);

    //对齐一下结构体，写的有点不太好
    if(rune){
        for(const auto& item : rune->results){

            Eigen::Vector2f t(item.points[0].x , item.points[0].y);
            Eigen::Vector2f l(item.points[1].x , item.points[1].y);
            Eigen::Vector2f b(item.points[4].x , item.points[4].y);
            Eigen::Vector2f r(item.points[3].x , item.points[3].y); //扇叶四个用于pnp的点
            float score = static_cast<float>(item.score);           //置信度

            buff_algo::BuffDetection detected(0 , item.class_id , score , t , l , b , r);
        //这里浙大的格式需要传入模型识别出的颜色，但是深大的模型没有考虑这个，所以直接传入我方颜色就ok
            detections.emplace_back(detected);
        };
    }; 

    //选择一个最近的扇叶来打
    std::vector<buff_algo::BuffDetection> selected;
    selector.select_buffs(detections, selected);

    //pnp解算 
    const std::vector<buff_algo::Pose3f> poses = solver.solve_pnp(selected);
    if (!poses.empty()){
        tracker.push(buff_algo::Buff(poses[0]));
    };

    //tracker更新
    const double timestamp = clip.video.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    tracker.update(timestamp); //时间戳，需改为同济形式

    buff_algo::BuffState state = tracker.get_state(); //拟合结果

    predictor.set_state(state, timestamp, timestamp);
    //auto x = predictor.predict_position(0.1f);

    //std::cout<<x<<std::endl;
    //打印调试信息
    if (tracker.status() != last_status || processed % 30 == 0) {
        tracker.print_colored_status_info();
        last_status = tracker.status();
    }
    ++processed;

    };
    return 0;
}
