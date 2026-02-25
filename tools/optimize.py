import optuna
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_slice,
    plot_parallel_coordinate,
)

from DetectionPipeline import DetectionConfig, DetectionPipeline
from DatasetLoader import load_dataset, calculate_error


def objective(trial):
    cfg = DetectionConfig(
        conf_thres=trial.suggest_float("conf_thres", 0.1, 0.8),
        iou_thres=trial.suggest_float("iou_thres", 0.1, 0.8),
    )

    pipeline = DetectionPipeline(config=cfg)
    total_error = 0
    for scenario in scenarios:
        result = pipeline.detect_frames(scenario.data_dir)
        error = calculate_error(result, scenario.ground_truth)
        total_error += error
    return total_error / len(scenarios)


if __name__ == "__main__":
    print("正在加载数据集...")
    scenarios = load_dataset("datasets/race")

    if not scenarios:
        raise RuntimeError("没有找到任何测试场景，请检查 datasets/race/ 目录！")

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)

    print("=======================================")
    print("最牛超参数:", study.best_params)
    print(f"在整个数据集上的最小误差总和: {study.best_value}")
    print("=======================================")

    bestCfg = DetectionConfig(**study.best_params)
    bestCfg.save_to_json("assets/best-param.json")

    print("正在生成可视化图表...")

    # 1. 优化历史图 (看误差是不是随着调参越来越小)
    fig_history = plot_optimization_history(study)
    fig_history.write_html("results/human/opt_history.html")

    # 2. 参数重要性图 (看 conf_thres 和 iou_thres 哪个对结果影响更大)
    fig_importance = plot_param_importances(study)
    fig_importance.write_html("results/human/opt_importance.html")

    # 3. 切片图 (看单个参数在什么区间表现最好，密集的谷底就是甜点区)
    fig_slice = plot_slice(study)
    fig_slice.write_html("results/human/opt_slice.html")

    # 4. 平行坐标图 (看高维参数之间的联动关系)
    fig_parallel = plot_parallel_coordinate(study)
    fig_parallel.write_html("results/human/opt_parallel.html")

    print("可视化完成！请双击打开 results/human/ 目录下的 .html 文件查看交互式图表。")
