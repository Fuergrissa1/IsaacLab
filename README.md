This repo can be extracted on top of the Isaac Lab directory without overwriting any files. 

From within your Isaac Lab virtual environment:

install python module (for skrl)  
```isaaclab.bat -i skrl```  
run script for training with the PPO algorithm  
```isaaclab.bat -p source\standalone\workflows\skrl\train.py --task Isaac-Tiltrotor-Direct-v0 --headless```  
run script for playing with 32 environments  
```isaaclab.bat -p source\standalone\workflows\skrl\play.py --task Isaac-Tiltrotor-Direct-v0 --num_envs 32 --checkpoint /PATH/TO/model.pt```  

## Tiltrotor 6D (Isaac Lab extension project)

The folder **`tiltrotor_6d_lab/`** is a separate Isaac Lab–style extension (manager-based env, skrl PPO, URDF under `tiltrotor_6d_lab/urdf/`). Install it in editable mode from that directory (`pip install -e source/tiltrotor_6d_lab`), then train e.g. `Template-Tiltrotor-6d-Hover-v0` per that project’s README. It lives in this repo alongside the 2D direct task under `IsaacLab/`.
