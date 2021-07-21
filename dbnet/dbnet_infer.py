import os
import onnxruntime as rt
import  numpy as np
import time
import cv2
from .decode import  SegDetectorRepresenter
import TopsInference

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def Singleton(cls):
    _instance = {}

    def _singleton(*args, **kargs):
        if cls not in _instance:
            _instance[cls] = cls(*args, **kargs)
        return _instance[cls]

    return _singleton


class SingletonType(type):
    def __init__(cls, *args, **kwargs):
        super(SingletonType, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls, *args, **kwargs)
        cls.__init__(obj, *args, **kwargs)
        return obj


def draw_bbox(img_path, result, color=(255, 0, 0), thickness=2):
    if isinstance(img_path, str):
        img_path = cv2.imread(img_path)
        # img_path = cv2.cvtColor(img_path, cv2.COLOR_BGR2RGB)
    img_path = img_path.copy()
    for point in result:
        point = point.astype(int)

        cv2.polylines(img_path, [point], True, color, thickness)
    return img_path


class DBNET(metaclass=SingletonType):
    def __init__(self, MODEL_PATH, run_mode, short_size):
        self.short_size = short_size
        self.engine_name_template = MODEL_PATH.replace('.onnx', '') + \
            ('_bs{{}}_h{{}}_w{{}}_{}.exec').\
            format(TopsInference.__version__.replace(' ', ''))
        self.engine_name = ""
        self.model_path = MODEL_PATH
        if run_mode is not "multiprocessing":
            self.handle = TopsInference.set_device(0, 0)
            self.engine = TopsInference.PyEngine()
        self.decode_handel = SegDetectorRepresenter()

    def process_mp(self, qimg, share_shape, share_buffer, qin, box_count):
        with TopsInference.device(0, 0):
            while True:
                try:
                    input_image = qimg.get()
                    # cv2.imshow("input image", input_image)
                    # cv2.waitKey(0)
                    share_shape[:] = list(input_image.shape)
                    buffer_len = share_shape[0] * share_shape[1] * share_shape[2]
                    share_buffer[:buffer_len] = list(input_image.reshape([-1,]))
                except BaseException:
                    print("get input image failed")
                    break
                img = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                if h < w:
                    scale_h = self.short_size / h
                    tar_w = w * scale_h
                    tar_w = tar_w - tar_w % 32
                    tar_w = max(32, tar_w)
                    scale_w = tar_w / w

                else:
                    scale_w = self.short_size / w
                    tar_h = h * scale_w
                    tar_h = tar_h - tar_h % 32
                    tar_h = max(32, tar_h)
                    scale_h = tar_h / h
                
                img = cv2.resize(img, None, fx=scale_w, fy=scale_h)
                img = img.astype(np.float32)
                img /= 255.0
                img -= mean
                img /= std
                img = img.transpose(2, 0, 1)
                transformed_image = np.expand_dims(img, axis=0)
                print("========================DBNET transformed_image.shape = {}".format(transformed_image.shape))
                engine_model_name = self.engine_name_template.format(
                    transformed_image.shape[0], transformed_image.shape[2], transformed_image.shape[3])
                if os.path.isfile(engine_model_name):
                    if self.engine_name == engine_model_name:
                        print("[DEBUG] Already load suitable enflame bin {}".format(engine_model_name))
                    else:
                        self.engine = TopsInference.load(engine_model_name)
                        self.engine_name = engine_model_name
                        print("Find engine file \'{}\'. Skip build engine.".format(
                            engine_model_name))
                else:
                    print("Fail to load model file:  {}".format(engine_model_name))
                    onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
                    onnx_parser.set_input_dtypes("DT_FLOAT32")
                    onnx_parser.set_input_shapes("1, 3, {}, {}".format(transformed_image.shape[2], transformed_image.shape[3]))
                    onnx_parser.set_input_names("input0")
                    onnx_parser.set_output_names("out1")
                    module = onnx_parser.read(self.model_path)
                    optimizer = TopsInference.create_optimizer()
                    print("build engine ...")
                    self.engine = optimizer.build(module)
                    print("build engine finished.")
                    self.engine.save_executable(engine_model_name)
                    self.engine_name = engine_model_name
                    print("save engine file: \'{}\'".format(engine_model_name))

                out = []
                self.engine.run([transformed_image.astype(np.float32, order='C')], out,
                            TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
                box_list, score_list = self.decode_handel(out[0][0], h, w)
                if len(box_list) > 0:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
                else:
                    box_list, score_list = [], []
                print("========================box_list.length = {}".format(len(box_list)))
                print("========================score_list.length = {}".format(len(score_list)))
                num_boxes = len(box_list)
                box_count.value = num_boxes
                sorted_index =  [x for x,y in sorted(enumerate(box_list), key = lambda x: (x[1][0][1], x[1][0][0]))]
                # sorted_boxes = sorted(box_list, key=lambda x: (x[0][1], x[0][0]))
                sorted_boxes = [box_list[i] for i in sorted_index]
                sorted_score = [score_list[i] for i in sorted_index]
                _boxes = list(sorted_boxes)
                print(_boxes)

                for i in range(num_boxes - 1):
                    if abs(_boxes[i+1][0][1] - _boxes[i][0][1]) < 10 and \
                        (_boxes[i + 1][0][0] < _boxes[i][0][0]):
                        tmp = _boxes[i]
                        _boxes[i] = _boxes[i + 1]
                        _boxes[i + 1] = tmp
                        tmp = sorted_score[i]
                        sorted_score[i] = sorted_score[i + 1]
                        sorted_score[i + 1] = tmp
                for count in range(num_boxes):
                        qin[count % 3].put([_boxes[count], sorted_score[count], count])

    def process(self, img, short_size):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h < w:
            scale_h = short_size / h
            tar_w = w * scale_h
            tar_w = tar_w - tar_w % 32
            tar_w = max(32, tar_w)
            scale_w = tar_w / w

        else:
            scale_w = short_size / w
            tar_h = h * scale_w
            tar_h = tar_h - tar_h % 32
            tar_h = max(32, tar_h)
            scale_h = tar_h / h
        


        img = cv2.resize(img, None, fx=scale_w, fy=scale_h)

        img = img.astype(np.float32)

        img /= 255.0
        img -= mean
        img /= std
        img = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(img, axis=0)
        print("========================DBNET transformed_image.shape = {}".format(transformed_image.shape))
        engine_model_name = self.engine_name_template.format(
            transformed_image.shape[0], transformed_image.shape[2], transformed_image.shape[3])
        if os.path.isfile(engine_model_name):
            if self.engine_name == engine_model_name:
                print("[DEBUG] Already load suitable enflame bin {}".format(engine_model_name))
            else:
                self.engine = TopsInference.load(engine_model_name)
                self.engine_name = engine_model_name
                print("Find engine file \'{}\'. Skip build engine.".format(
                    engine_model_name))
        else:
            print("Fail to load model file:  {}".format(engine_model_name))
            onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
            onnx_parser.set_input_dtypes("DT_FLOAT32")
            onnx_parser.set_input_shapes("1, 3, {}, {}".format(transformed_image.shape[2], transformed_image.shape[3]))
            onnx_parser.set_input_names("input0")
            onnx_parser.set_output_names("out1")
            module = onnx_parser.read(self.model_path)
            optimizer = TopsInference.create_optimizer()
            print("build engine ...")
            self.engine = optimizer.build(module)
            print("build engine finished.")
            self.engine.save_executable(engine_model_name)
            self.engine_name = engine_model_name
            print("save engine file: \'{}\'".format(engine_model_name))

        out = []
        self.engine.run([transformed_image.astype(np.float32, order='C')], out,
                       TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
        box_list, score_list = self.decode_handel(out[0][0], h, w)
        if len(box_list) > 0:
            idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
            box_list, score_list = box_list[idx], score_list[idx]
        else:
            box_list, score_list = [], []
        print("========================box_list.length = {}".format(len(box_list)))
        # print(box_list)
        print("========================score_list.length = {}".format(len(score_list)))
        return box_list, score_list


if __name__ == "__main__":
    text_handle = DBNET(MODEL_PATH="./model/dbnet.onnx")
    img = cv2.imread("../test_imgs/1.jpg")
    print(img.shape)
    box_list, score_list = text_handle.process(img)
    img = draw_bbox(img, box_list)
    cv2.imwrite("test.jpg", img)
