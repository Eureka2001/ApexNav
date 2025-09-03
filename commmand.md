```bash
# 开启 FTXUI（可能需要指定 path）
rosrun state_monitor state_monitor_node _previous_record_path:=./videos/test_hm3dv1_val/record.txt
rosrun state_monitor state_monitor_node _previous_record_path:=./videos/test_hm3dv2_val/record.txt
```

```bash
python -m vlm.detector.grounding_dino
python -m vlm.itm.blip2itm
python -m vlm.segmentor.sam
python -m vlm.detector.yolov7
```

```bash
source ./devel/setup.bash && roslaunch exploration_manager exploration.launch
```

```bash
source ./devel/setup.bash
python habitat_evaluation.py --dataset hm3dv1
python habitat_evaluation.py --dataset hm3dv2
python habitat_evaluation.py --dataset mp3d
python habitat_evaluation.py --dataset hm3dv2 test_epi_num=10 need_video=true
```