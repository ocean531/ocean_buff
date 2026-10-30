#include "main.hpp"


int main(int argc, char **argv)
{
    const fs::path root = projectRoot(argc, argv);
    const fs::path config = argc > 2
                                ? fs::absolute(fs::path(argv[2]))
                                : root / "configs/detect_config.json";

    const fs::path output_dir = root / "result";

    std::optional<RuneDetectionFrame> rune = runRune(root, config, output_dir, 0);

    return 0;
}
