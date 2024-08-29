### Robotic Object Insertion with a Soft Wrist through Sim-to-Real Privileged Training (IROS2024)

## setup
1. install docker
```bash
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```
2. create a parent folder and move docker-related files to the parent folder. The file architecture should look like following
```
- softrobot-teacher-student-training/
  - build_docker.sh
  - Dockerfile
  - exec_docker.sh
  - run_docker.sh
  - robosuite/
    - [files in the repository]
```
3. allow containers to connect to the local X server
```bash
sudo xhost + local:docker
```
4. build and execute docker
```bash
build_docker.sh
run_docker.sh
```
## Training

Teacher policy
```bash
cd /robosuite/robosuite/learning/
python3 -u train_policy.py -n ${directory_name} --param1 $param1 --param2 $param2 --seed $seed
```

param1 is hole position variation
param2 is peg angle variation

Please select the state in `/robosuite/robosuite/environments/manipulation/soft_peg_in_hole.py` as follows.
```python
sensors = [
            wrist_pos_rel,
            # wrist_rot6d,
            # wrist_wrench,
            wrist_force,
            # peg_torque,
            # spring_angle,
            peg_pos_rel,
            peg_rot6d,
            # peg_vel,
            # wrist_vel,
            # peg_omega,
            # wrist_omega,
            # time_phase,
            # joint_gains,
            # hole_offset,
            # peg_angle,
            peg_alignment,
        ]
        names = [
            "wrist_pos_rel",
            # "wrist_rot6d",
            # "wrist_wrench",
            "wrist_force",
            # "peg_torque",
            # "spring_angle",
            # "peg_pos_rel",
            "peg_rot6d",
            "peg_vel",
            # "wrist_vel",
            # "peg_omega",
            # "wrist_omega",
            # "time_phase",
            # "joint_gains",
            # "hole_offset",
            # "peg_angle",
            "peg_alignment",
        ]

```

Student encoder
```bash
cd /robosuite/robosuite/learning/
python3 -u train_student.py learning_progress/${model_name.pt} -n ${directory name} --param1 $param1 --param2 $param2
```

Please select the state in `/robosuite/robosuite/environments/manipulation/soft_peg_in_hole.py` as follows.

```python
sensors = [
            wrist_pos_rel,
            # wrist_rot6d,
            # wrist_wrench,
            wrist_force,
            # peg_torque,
            # spring_angle,
            # peg_pos_rel,
            # peg_rot6d,
            # peg_vel,
            # wrist_vel,
            # peg_omega,
            # wrist_omega,
            # time_phase,
            # joint_gains,
            # hole_offset,
            # peg_angle,
            # peg_alignment,
        ]
        names = [
            "wrist_pos_rel",
            # "wrist_rot6d",
            # "wrist_wrench",
            "wrist_force",
            # "peg_torque",
            # "spring_angle",
            # "peg_pos_rel",
            # "peg_rot6d",
            # "peg_vel",
            # "wrist_vel",
            # "peg_omega",
            # "wrist_omega",
            # "time_phase",
            # "joint_gains",
            # "hole_offset",
            # "peg_angle",
            # "peg_alignment",
        ]
```

Peg is defined in `/robosuite_ws/robosuite/robosuite/models/assets/grippers/robotiq_gripper_hand_e_soft.xmlH`
```xml
  <!-- peg visual and collision geometries in the order of: circle, square, triangle, rectangle-->
  <!-- uncomment the two subsequent lines for the desired peg type -->
  <geom type="cylinder" name="peg_visual" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.045" contype="0" conaffinity="0" group="1" rgba="0 0.7 0.7 1"/>
  <geom type="cylinder" name="peg_collision" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.045" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4"/>
  <!-- <geom type="box" name="peg_visual" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.02 0.045" contype="0" conaffinity="0" group="1" rgba="0 0.7 0.7 1"/>
  <geom type="box" name="peg_collision" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.02 0.045" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4"/> -->
  <!-- <geom type="mesh" name="peg_visual" mesh="guriguri-large-plate-peg" pos="0 0 0.055" quat="0 0 1 0" contype="0" conaffinity="0" group="1" rgba="0 0.7 0.7 1"/>
  <geom type="mesh" name="peg_collision" mesh="guriguri-large-plate-peg-collision" pos="0 0 0.055" quat="0 1 0 0" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4"/> -->
  <!-- <geom type="box" name="peg_visual" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.01 0.045" contype="0" conaffinity="0" group="1" rgba="0 0.7 0.7 1"/>
  <geom type="box" name="peg_collision" pos="0.0 0.0 0.04" quat="0 1 0 0" size="0.02 0.01 0.045" group="0" conaffinity="1" contype="0" solref="0.02 1" friction="1 0.005 0.0001" condim="4"/> -->
```

Hole is defined in soft_peh_in_hole.py
```python
  self.hole = GuriguriLargeRoundHoleObject(name="hole")
  # self.hole = GuriguriLargeSquareHoleObject(name="hole")
  # self.hole = GuriguriLargeTriangleHoleObject(name="hole")
  # self.hole = GuriguriLargeRectangleHoleObject(name="hole")
```

#Test
```bash
cd /robosuite/robosuite/learning/
```

Test teacher policy
```bash
python3 test_policy.py
```
Test student policy
```bash
python3 test_student.py
```

Our proposed method is based on Robosuite. https://github.com/ARISE-Initiative/robosuite