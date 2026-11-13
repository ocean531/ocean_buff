#include "main.hpp"


int main(int argc, char **argv)
{

    buff_algo::BuffSelector selector(buff_color); //扇叶选择器
    selector.set_camera_matrix(kCameraMatrix, kDistortion); //后面改成从yaml中读取

    buff_algo::BuffPnPSolver solver; //解算器
    solver.set_camera_matrix(kCameraMatrix, kDistortion);//同上,需要修改

    buff_algo::BuffTracker tracker(tracker_config["buff_tracker"]);
    tracker.set_mode(static_cast<buff_algo::BuffMode>(mode));

    buff_algo::BuffPredictor predictor(predictor_config["buff_predictor"]);




    const fs::path root = projectRoot(argc, argv);
    const fs::path config = argc > 2
                                ? fs::absolute(fs::path(argv[2]))
                                : root / "configs/detect_config.json";

    const fs::path output_dir = root / "result";

    cv::Mat frame;
    int processed = 0;
    bool window_ok = true;
    buff_algo::StatusType last_status = buff_algo::StatusType::LOST;

    while(1){ //主循环入口

    std::optional<RuneDetectionFrame> rune = runRune(root, config, output_dir, 0); //模型处理结果

    std::vector<buff_algo::BuffDetection> detections ;//算法需要的类型
    detections.reserve(100000);

    //对齐一下结构体，写的有点不太好
    if(rune){
        for(const auto& item : rune->results){

            Eigen::Vector2f t(item.points[0].x , item.points[0].y);
            Eigen::Vector2f l(item.points[1].x , item.points[1].y);
            Eigen::Vector2f b(item.points[4].x , item.points[4].y);
            Eigen::Vector2f r(item.points[3].x , item.points[3].y); //扇叶四个用于pnp的点
            float score = static_cast<float>(item.score);           //置信度

            buff_algo::BuffDetection detected(0 , item.class_id , score , t , l , b , r);
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
    const double timestamp = capture.get(cv::CAP_PROP_POS_MSEC) / 1000.0;
    tracker.update(timestamp); //时间戳，需改为同济形式

    buff_algo::BuffState state = tracker.get_state(); //拟合结果

    predictor.set_state(state, timestamp, timestamp);
    predictor.predict_position(0.1f);

    //打印调试信息
    if (tracker.status() != last_status || processed % 30 == 0) {
        tracker.print_colored_status_info();
        last_status = tracker.status();
    }



    }
    return 0;
}
