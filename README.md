# live_cam_changer
change face in real time 

python3 setup/setup.py --no-gpu-check --config setup/config.yaml

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
