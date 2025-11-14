conda activate base

atc --framework=5 \
    --model="/home/HwHiAiUser/Desktop/raceone/model/overall.onnx" \
    --input_format=NCHW \
    --input_shape="images:1,3,480,640" \
    --output="/home/HwHiAiUser/Desktop/raceone/model/overall" \
    --soc_version="Ascend310B4" \
    # --log=error \
    # --precision_mode=force_fp32 \

