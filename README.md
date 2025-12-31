# live_cam_changer
change face in real time 

python3 setup/setup.py --no-gpu-check --skip-convert --config setup/config.yaml

python3 setup/setup.py  

python3 setup/update_repo.py


check
project-root/
├── setup/
│   └── setup.py              # manual setup script
├── models/
│   ├── onnx/
│   │   └── face_det.onnx
│   ├── tensorrt/
│   │   └── face_det.trt
├── src/
│   ├── main.cpp              # C++ face detection
│   └── CMakeLists.txt
├── scripts/
│   └── run.sh
├── README.md
