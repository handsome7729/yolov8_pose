


'''此代码为备份'''

import argparse
import time
from abc import abstractmethod, ABC

import numpy
import numpy as np
import torch

import util2 as util
import cv2
import acl

DEVICE_ID = 0  # 设备id
SUCCESS = 0  # 成功状态值
FAILED = 1  # 失败状态值
ACL_MEM_MALLOC_NORMAL_ONLY = 2  # 申请内存策略, 仅申请普通页

def init_acl(device_id):
    acl.init()
    ret = acl.rt.set_device(device_id)  # 指定运算的Device
    if ret:
        raise RuntimeError(ret)
    context, ret = acl.rt.create_context(device_id)  # 显式创建一个Context
    if ret:
        raise RuntimeError(ret)
    print('Init ACL Successfully')
    return context


def deinit_acl(context, device_id):
    ret = acl.rt.destroy_context(context)  # 释放 Context
    if ret:
        raise RuntimeError(ret)
    ret = acl.rt.reset_device(device_id)  # 释放Device
    if ret:
        raise RuntimeError(ret)
    ret = acl.finalize()  # 去初始化
    if ret:
        raise RuntimeError(ret)
    print('Deinit ACL Successfully')


class Model(ABC):
    def __init__(self, model_path):
        print(f"load model {model_path}")
        self.model_path = model_path  # 模型路径
        self.model_id = None  # 模型 id
        self.input_dataset = None  # 输入数据结构
        self.output_dataset = None  # 输出数据结构
        self.model_desc = None  # 模型描述信息
        self._input_num = 0  # 输入数据个数
        self._output_num = 0  # 输出数据个数
        self._output_info = []  # 输出信息列表
        self._is_released = False  # 资源是否被释放
        self._init_resource()

    def _init_resource(self):
        ''' 初始化模型、输出相关资源。相关数据类型: aclmdlDesc aclDataBuffer aclmdlDataset'''
        print("Init model resource")
        # 加载模型文件
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)  # 加载模型
        self.model_desc = acl.mdl.create_desc()  # 初始化模型信息对象
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)  # 根据模型获取描述信息
        print("[Model] Model init resource stage success")

        # 创建模型输出 dataset 结构
        self._gen_output_dataset()  # 创建模型输出dataset结构

    def _gen_output_dataset(self):
        ''' 组织输出数据的dataset结构 '''
        ret = SUCCESS
        self._output_num = acl.mdl.get_num_outputs(self.model_desc)  # 获取模型输出个数
        self.output_dataset = acl.mdl.create_dataset()  # 创建输出dataset结构
        for i in range(self._output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)  # 获取模型输出个数
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)  # 为每个输出申请device内存
            dataset_buffer = acl.create_data_buffer(temp_buffer,
                                                    temp_buffer_size)  # 创建输出的data buffer结构,将申请的内存填入data buffer
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, dataset_buffer)  # 将 data buffer 加入输出dataset

        if ret == FAILED:
            self._release_dataset(self.output_dataset)  # 失败时释放dataset
        # print("[Model] create model output dataset success")

    def _gen_input_dataset(self, input_list):
        ''' 组织输入数据的dataset结构 '''
        ret = SUCCESS
        self._input_num = acl.mdl.get_num_inputs(self.model_desc)  # 获取模型输入个数
        self.input_dataset = acl.mdl.create_dataset()  # 创建输入dataset结构
        for i in range(self._input_num):
            item = input_list[i]  # 获取第 i 个输入数据
            data_ptr = acl.util.bytes_to_ptr(item.tobytes())  # 获取输入数据字节流
            size = item.size * item.itemsize  # 获取输入数据字节数
            dataset_buffer = acl.create_data_buffer(data_ptr, size)  # 创建输入dataset buffer结构, 填入输入数据
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, dataset_buffer)  # 将dataset buffer加入dataset

        if ret == FAILED:
            self._release_dataset(self.input_dataset)  # 失败时释放dataset
        # print("[Model] create model input dataset success")

    def _unpack_bytes_array(self, byte_array, shape, datatype):
        ''' 将内存不同类型的数据解码为numpy数组 '''
        np_type = None

        # 获取输出数据类型对应的numpy数组类型和解码标记
        if datatype == 0:  # ACL_FLOAT
            np_type = np.float32
        elif datatype == 1:  # ACL_FLOAT16
            np_type = np.float16
        elif datatype == 3:  # ACL_INT32
            np_type = np.int32
        elif datatype == 8:  # ACL_UINT32
            np_type = np.uint32
        else:
            print("unsurpport datatype ", datatype)
            return

        # 将解码后的数据组织为numpy数组,并设置shape和类型
        return np.frombuffer(byte_array, dtype=np_type).reshape(shape)

    def _output_dataset_to_numpy(self):
        ''' 将模型输出解码为numpy数组 '''
        dataset = []
        # 遍历每个输出
        for i in range(self._output_num):
            buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)  # 从输出dataset中获取buffer
            data_ptr = acl.get_data_buffer_addr(buffer)  # 获取输出数据内存地址
            size = acl.get_data_buffer_size(buffer)  # 获取输出数据字节数
            narray = acl.util.ptr_to_bytes(data_ptr, size)  # 将指针转为字节流数据

            # 根据模型输出的shape和数据类型,将内存数据解码为numpy数组
            dims = acl.mdl.get_output_dims(self.model_desc, i)[0]["dims"]  # 获取每个输出的维度
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)  # 获取每个输出的数据类型
            output_nparray = self._unpack_bytes_array(narray, tuple(dims), datatype)  # 解码为numpy数组
            dataset.append(output_nparray)
        return dataset

    def execute(self, input_list):
        '''创建输入dataset对象, 推理完成后, 将输出数据转换为numpy格式'''
        self._gen_input_dataset(input_list)  # 创建模型输入dataset结构
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)  # 调用离线模型的execute推理数据
        out_numpy = self._output_dataset_to_numpy()  # 将推理输出的二进制数据流解码为numpy数组, 数组的shape和类型与模型输出规格一致
        return out_numpy

    def release(self):
        ''' 释放模型相关资源 '''
        if self._is_released:
            return

        print("Model start release...")
        self._release_dataset(self.input_dataset)  # 释放输入数据结构
        self.input_dataset = None  # 将输入数据置空
        self._release_dataset(self.output_dataset)  # 释放输出数据结构
        self.output_dataset = None  # 将输出数据置空

        if self.model_id:
            ret = acl.mdl.unload(self.model_id)  # 卸载模型
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)  # 释放模型描述信息
        self._is_released = True
        print("Model release source success")

    def _release_dataset(self, dataset):
        ''' 释放 aclmdlDataset 类型数据 '''
        if not dataset:
            return
        num = acl.mdl.get_dataset_num_buffers(dataset)  # 获取数据集包含的buffer个数
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)  # 获取buffer指针
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)  # 释放buffer
        ret = acl.mdl.destroy_dataset(dataset)  # 销毁数据集

    @abstractmethod
    def infer(self, inputs):  # 保留接口, 子类必须重写
        pass


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


class Pose(Model):
    def __init__(self, modelpath, input_size):
        super().__init__(modelpath)
        self.input_size = input_size

    @torch.no_grad()
    def infer(self):

        palette = numpy.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                               [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                               [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                               [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                              dtype=numpy.uint8)
        skeleton = [[16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
                    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7]]
        kpt_color = palette[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
        limb_color = palette[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
        stride = 32

        # 初始化视频流
        camera = cv2.VideoCapture('./data/test.mp4')
        # 获取视频帧的宽度和高度
        frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #print("frame width",frame_width)
        #hph2.mp4:frame width 544 frame height 960
           
        #print("frame height",frame_height)
        # Check if camera opened successfully
        if not camera.isOpened():
            print("Error opening video stream or file")

        out = cv2.VideoWriter('./data/output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                              (frame_width, frame_height))

        # Read until video is completed
        while camera.isOpened():
            # Capture frame-by-frame
            success, frame = camera.read()
            if success:
	# 前处理
                image = frame.copy()
                resize_image=frame.copy()
                shape = image.shape[:2]  # current shape [height, width]
                print("shape",shape)
                image = cv2.resize(image, (self.input_size, self.input_size))
                resize_image = cv2.resize(resize_image, (self.input_size, self.input_size))

                image = cv2.dnn.blobFromImage(image, scalefactor=1 / 255.0, swapRB=True)

                # 推理
                t1 = time.time()
                outputs = self.execute([image, ])[0]
                #print("outputs",outputs)
                #print("shape",outputs.shape)
                print("infer time",time.time() - t1)

	# 后处理
                # NMS
                outputs = torch.from_numpy(np.array(outputs, copy=True)) 
                outputs = util.non_max_suppression(outputs, 0.25, 0.7, 1)
                for output in outputs:
                    output = output.clone()
                    #print("output:",output)# 前6个数字是人物框 x1 y1 x2 y2 confidence class，后面17*3个数字是17个骨架关键点
                    if len(output):
                        box_output = output[:, :6]
                        #print("box_output",box_output) #人物框 x1 y1 x2 y2 confidence class
                        kps_output = output[:, 6:].view(len(output), 17, 3)
                        #print("kps_out",kps_output)#骨架 17组 x y confidence
                    else:
                        box_output = output[:, :6]
                        kps_output = output[:, 6:]
                    #print("image.shape:",image.shape)#image.shape: (1, 3, 640, 640)
                    #r = min(image.shape[2] / shape[0], image.shape[3] / shape[1])
                    #shape[0]为原始高度，shape[1]为原始宽度
                
                    # 计算缩放比例
                    scale_w = shape[1] / self.input_size  # 宽度缩放比例
                    scale_h = shape[0] / self.input_size  # 高度缩放比例
                    # 遍历边界框
                    for box in box_output:
                        #box = box.cpu().numpy()
                        x1, y1, x2, y2, score, cls = box.cpu().numpy()
                        x1, x2 = x1 * scale_w, x2 * scale_w
                        y1, y2 = y1 * scale_h, y2 * scale_h
                        #x1, y1, x2, y2, score, index = box
                        #为了验证模型，把画骨架的图改为resize后的
                        cv2.rectangle(frame,
                                      (int(x1), int(y1)),
                                      (int(x2), int(y2)),
                                      (0, 255, 0), 2)
                    # 遍历关键点
                    
                    for kpt in reversed(kps_output):
                        for i, k in enumerate(kpt):
                            color_k = [int(x) for x in kpt_color[i]]
                            x_coord, y_coord = k[0], k[1]
                            x_coord, y_coord = x_coord * scale_w, y_coord * scale_h
                            if x_coord % shape[1] != 0 and y_coord % shape[0] != 0:
                                if len(k) == 3:
                                    conf = k[2]
                                    if conf < 0.5:
                                        continue
                                cv2.circle(frame,
                                           (int(x_coord), int(y_coord)),
                                           5, color_k, -1, lineType=cv2.LINE_AA)
                        # 遍历骨架结构                   
                        for i, sk in enumerate(skeleton):
                            pos1 = (int(kpt[(sk[0] - 1), 0]* scale_w), int(kpt[(sk[0] - 1), 1]* scale_h))
                            pos2 = (int(kpt[(sk[1] - 1), 0]* scale_w), int(kpt[(sk[1] - 1), 1]* scale_h))
                            if kpt.shape[-1] == 3:
                                conf1 = kpt[(sk[0] - 1), 2]
                                conf2 = kpt[(sk[1] - 1), 2]
                                if conf1 < 0.5 or conf2 < 0.5:
                                    continue
                            if pos1[0] % shape[1] == 0 or pos1[1] % shape[0] == 0 or pos1[0] < 0 or pos1[1] < 0:
                                continue
                            if pos2[0] % shape[1] == 0 or pos2[1] % shape[0] == 0 or pos2[0] < 0 or pos2[1] < 0:
                                continue
                            cv2.line(frame,
                                     pos1, pos2,
                                     [int(x) for x in limb_color[i]],
                                     thickness=2, lineType=cv2.LINE_AA)
                #print(image.dtype)
                cv2.imshow("img",frame)
                cv2.waitKey(5)
                out.write(frame)
            else:
                break
        camera.release()
        out.release()


context = init_acl(DEVICE_ID)  # 初始化acl相关资源

pose = Pose(modelpath='weights/v8_n_pose.om' ,input_size=640)

pose.infer()

pose.release()  # 释放acl模型相关资源, 包括输入数据、输出数据、模型等
deinit_acl(context, 0)  # acl去初始化
