```bash
# 开启 FTXUI（需要指定不同的 path）
rosrun state_monitor state_monitor_node _previous_record_path:=./videos/test_hm3dv1_val/record.txt
rosrun state_monitor state_monitor_node _previous_record_path:=./videos/test_hm3dv2_val/record.txt
rosrun state_monitor state_monitor_node _previous_record_path:=./videos/test_mp3d_val/record.txt

# 开启 RVIZ
source ./devel/setup.bash && roslaunch exploration_manager rviz.launch
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

```bash
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
```

```bash
python habitat_manual_control.py --dataset hm3dv1 test_epi_num=10
```
