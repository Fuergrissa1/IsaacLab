This repo can be extracted on top of the Isaac Lab directory without overwriting any files. 

From within your Isaac Lab virtual environment:

install python module (for skrl)

`isaaclab.bat -i skrl`
run script for training with the PPO algorithm

`isaaclab.bat -p source\standalone\workflows\skrl\train.py --task Isaac-Tiltrotor-Direct-v0 --headless`
run script for playing with 32 environments 

`isaaclab.bat -p source\standalone\workflows\skrl\play.py --task Isaac-Tiltrotor-Direct-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt`
