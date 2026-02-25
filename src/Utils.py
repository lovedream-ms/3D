from collections import Counter
from typing import List

from Config import resultPath


def _write_txt(results: List[Counter]):
    """
    写单轮（或单桌）结果到 result_r，格式：
    START
    Goal_ID=CA001;Num=2;Table=1
    Goal_ID=CA002;Num=1;Table=3
    END
    """
    lines = []

    lines.append("START")
    for tableIndex, result in enumerate(results, start=1):
        for k, v in result.items():
            lines.append(f"Goal_ID={k};Num={int(v)};Table={int(tableIndex)}")
    lines.append("END")

    with open(resultPath, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def initResult():
    import os

    os.makedirs("results/human/camera", exist_ok=True)
    for num in range(3):
        os.makedirs(f"results/machine/camera/T{num+1}", exist_ok=True)
        os.makedirs(f"results/machine/detection/T{num+1}", exist_ok=True)


if __name__ == "__main__":
    # 测试写文件
    testResults = [Counter({"CA001": 2, "CA002": 1}), Counter({"CA001": 1})]
    _write_txt(testResults)
