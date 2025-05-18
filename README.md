# yolov8_pose
基于香橙派ai pro进行yolov8人体姿态检测

## 下载模型，我这里选择的是原始的yolov8n-pose.pt，这个模型应该网上可以下载到

## 进行模型转换：pt模型转换成onnx模型,参考pt2onnx.py

## onnx模型要转换成om模型
我主要参考这个项目里的教程：https://gitee.com/ascend/EdgeAndRobotics/tree/master/Samples/YOLOV5MultiInput#https://gitee.com/link?target=https%3A%2F%2Fhiascend.com%2Fdocument%2Fredirect%2FCannCommunityAtc

### 设置环境变量，配置程序编译依赖的头文件与库文件路径

export DDK_PATH=/usr/local/Ascend/ascend-toolkit/latest 

export NPU_HOST_LIB=$DDK_PATH/runtime/lib64/stub

当设备内存小于8G时，可设置如下两个环境变量减少atc模型转换过程中使用的进程数，减小内存占用。

export TE_PARALLEL_COMPILER=1

export MAX_COMPILE_CORE_NUMBER=1

atc --model=yolov8n-pose.onnx --framework=5 --output=yolov8n-pose --input_shape="images:1,3,640,640;img_info:1,4"  --soc_version=Ascend310B4 --insert_op_conf=aipp.cfg

--model：Resnet50网络模型文件所在路径。

--framework：原始框架类型，5表示ONNX。
--output：生成的离线模型路径。

--input_shape：模型输入数据的shape。这里我忘记我这个模型的input shape是多少了，需要再手动查查，建议使用[netron](http://netron.app "netron")网站可视化查看

--soc_version：昇腾AI处理器的型号。香橙派ai pro是Ascend310B4

一切顺利的话，等待20-30min就转换成功了


需要自己修改输入的视频，主要代码在infer()里面
![res](https://github.com/user-attachments/assets/e4dc2161-73d6-475b-9f0e-d6e59ebce763)
