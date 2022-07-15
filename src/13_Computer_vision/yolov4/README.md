# libtorch yolov4

yolov4 implement by libtorch in c++

[Get yolov4.cfg and yolov4.weights from] (https://github.com/AlexeyAB/darknet)

[Download yolov4.cfg](https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg)
[Download yolov4.weights](https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights)

### LibTorch version

LibTorch 1.11.0+cpu

### usage

can only inference by now

```
git clone
mkdir build
cmake -DCMAKE_PREFIX_PATH=<libtorch abs path> ..
./yolov4 <yolov4.cfg> <yolov4.weights> <image_path>
```

the result write to det_result.png

[libtorch-yolov3](https://github.com/walktree/libtorch-yolov3)
[pytorch-YOLOV4](https://github.com/Tianxiaomo/pytorch-YOLOv4)


