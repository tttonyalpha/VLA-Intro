conda create -y -n workshop_env python=3.12
conda activate workshop_env


pip install lerobot
pip install 'lerobot[feetech]'
pip install pyrealsense2

(Если rerun не будет видеть init)
python -m pip uninstall -y rerun rerun-sdk
python -m pip install "rerun-sdk>=0.24.0,<0.27.0"

https://huggingface.co/MrAnton