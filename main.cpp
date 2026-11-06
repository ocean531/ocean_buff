#include "main.hpp"


int main(int argc, char **argv)
{

    buff_algo::BuffSelector selector(buff_color); //扇叶选择器
    selector.set_camera_matrix(kCameraMatrix, kDistortion); //后面改成从yaml中读取







    const fs::path root = projectRoot(argc, argv);
    const fs::path config = argc > 2
                                ? fs::absolute(fs::path(argv[2]))
                                : root / "configs/detect_config.json";

    const fs::path output_dir = root / "result";

    std::optional<RuneDetectionFrame> rune = runRune(root, config, output_dir, 0); //模型处理结果
    std::vector<buff_algo::BuffDetection> detections ;//算法需要的类型
    detections.reserve(100000);

    //对齐一下结构体，写的有点难受
    if(rune){
        for(const auto& item : rune->results){

            Eigen::Vector2f t(item.points[0].x , item.points[0].y);
            Eigen::Vector2f l(item.points[1].x , item.points[1].y);
            Eigen::Vector2f b(item.points[4].x , item.points[4].y);
            Eigen::Vector2f r(item.points[3].x , item.points[3].y);
            float score = static_cast<float>(item.score);

            buff_algo::BuffDetection detected(0 , item.class_id , score , t , l , b , r);
            detections.emplace_back(detected);
        };
    }; 

    std::vector<buff_algo::BuffDetection> selected;
    selector.select_buffs(detections, selected);

    return 0;
}
