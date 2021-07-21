import queue
from config import *
from crnn import CRNNHandle
from angnet import  AngleNetHandle
from utils import draw_bbox, crop_rect, sorted_boxes, get_rotate_crop_image
from PIL import Image
import numpy as np
from cv2 import cv2
import copy
from dbnet.dbnet_infer import DBNET
import time
import traceback
import multiprocessing
import TopsInference

class  OcrHandle(object):
    def __init__(self):
        self.text_handle = DBNET(dbnet_model_path, run_mode, short_size)
        self.crnn_handle = CRNNHandle(crnn_model_path, run_mode)
        if angle_detect:
            self.angle_handle = AngleNetHandle(angle_net_path, run_mode)
        
        # with TopsInference.device(0, 3):
        #     for i in range(7):
        #         engine_model_name = crnn_model_path.replace('.onnx', '') + \
        #             ('_bs1_h32_w{}_{}.exec').\
        #             format((i + 1) *  crnn_width_step, 
        #             TopsInference.__version__.replace(' ', ''))
        #         print("current model is {}".format(engine_model_name))
        #         if os.path.isfile(engine_model_name):
        #             print("engine_model_name already exist")
        #             continue
        #         else:
        #             print("engine_model_name build")
        #             onnx_parser = TopsInference.create_parser(TopsInference.ONNX_MODEL)
        #             onnx_parser.set_input_dtypes("DT_FLOAT32")
        #             onnx_parser.set_input_shapes("1, 3, 32, {}".format((i + 1) *  crnn_width_step))
        #             onnx_parser.set_input_names("input")
        #             onnx_parser.set_output_names("out")
        #             module = onnx_parser.read(crnn_model_path)
        #             optimizer = TopsInference.create_optimizer()
        #             print("build engine ...")
        #             eigine = optimizer.build(module)
        #             print("build engine finished.")
        #             eigine.save_executable(engine_model_name)
        #             print("save engine file: \'{}\'".format(engine_model_name))

        if run_mode == "multiprocessing":
            # input data
            self.queue_input = multiprocessing.Queue(maxsize=1)
            self.qin_0 = multiprocessing.Queue(maxsize=1)
            self.qin_1 = multiprocessing.Queue(maxsize=1)
            self.qin_2 = multiprocessing.Queue(maxsize=1)
            self.qin_3 = multiprocessing.Queue(maxsize=1)
            self.queues_in = [self.qin_0, self.qin_1, self.qin_2, self.qin_3]
            self.qout_0 = multiprocessing.Queue(maxsize=5)
            self.qout_1 = multiprocessing.Queue(maxsize=5)
            self.qout_2 = multiprocessing.Queue(maxsize=5)
            self.qout_3 = multiprocessing.Queue(maxsize=5)
            self.queues_out = [self.qout_0, self.qout_1, self.qout_2, self.qout_3]
            self.qin_box = multiprocessing.Queue(maxsize=1)
            self.box_count = multiprocessing.Value("i", -1)
            self.qoutput = multiprocessing.Queue(maxsize=1)
            
            self.text_process = multiprocessing.Process(
                target=self.text_handle.process_mp,
                args=(
                    self.queue_input,
                    self.queues_in, 
                    self.box_count),
                name="text_detect")
            self.text_process.start()
            self.text_process_list = []
            for i in range(3):
                self.text_process_list.append( 
                    multiprocessing.Process(
                    target=self.crnnRecWithBox_mp,
                    args=(
                        i,
                        self.queues_in[i], 
                        self.queues_out[i],
                        self.box_count),
                    name="text_recog_{}".format(i)) )
                self.text_process_list[i].start()
            self.gather_process = multiprocessing.Process(
                target=self.gather_result_mp,
                args=(
                    self.queues_out,
                    self.qoutput,
                    self.box_count),
                name="post_process")
            self.gather_process.start()

    def gather_result_mp(self, queues_out, qoutput, box_count):
        results = []
        count = 0
        while True:
            for i in range(3):
                try:
                    cur_result = queues_out[i].get()
                except BaseException:
                    return
                results.append(cur_result)
                count = count + 1
                if count == box_count.value:
                    qoutput.put(results)
                    results = []
                    count = 0
                    break


    def crnnRecWithBox_mp(self, index, qin, qout, box_count):
        with TopsInference.device(0, index + 1):
            while True:
                try:
                    cur_box, cur_socre, count, input_img = qin.get()
                except BaseException:
                    print("cluster id:{} exit".format(index + 1))
                    break
                results = []
                tmp_box = copy.deepcopy(cur_box)
                partImg_array = get_rotate_crop_image(input_img, tmp_box.astype(np.float32))
                print("partImg_array.shape = {}".format(partImg_array.shape))
                partImg = Image.fromarray(partImg_array).convert("RGB")
                scale = partImg.size[1] * 1.0 / box_standard_height
                w = partImg.size[0] / scale
                w = int(w)
                img = partImg.resize((w, box_standard_height), Image.BILINEAR)
                angle_res = False
                if angle_detect:
                    if w < angle_detect_weight:
                        imgnew = Image.new('RGB', (angle_detect_weight, box_standard_height), (255))
                        imgnew.paste(img, (0, 0, w, box_standard_height))
                    else :
                        imgnew = img.crop((0, 0, angle_detect_weight, box_standard_height))
                    angle_detect_img = np.array(imgnew, dtype=np.float32)
                    angle_detect_img -= 127.5
                    angle_detect_img /= 127.5
                    angle_detect_image = angle_detect_img.transpose(2, 0, 1)
                    angle_detect_transformed_image = np.expand_dims(angle_detect_image, axis=0)
                    angle_res = self.angle_handle.predict_rbg_mp(angle_detect_transformed_image)
                if angle_detect and angle_res:
                    img = img.rotate(180)
                crnn_width = (w // crnn_width_step ++ 1) * crnn_width_step
                crnn_imgnew = Image.new('RGB', (crnn_width, box_standard_height), (255))
                crnn_imgnew.paste(img, (0, 0, w, box_standard_height))
                crnn_img = np.array(crnn_imgnew, dtype=np.float32)
                crnn_img -= 127.5
                crnn_img /= 127.5
                crnn_image = crnn_img.transpose(2, 0, 1)
                crnn_transformed_image = np.expand_dims(crnn_image, axis=0)
                simPred = self.crnn_handle.predict_rbg_mp(crnn_transformed_image)
                if simPred.strip() != '':
                    results = [tmp_box,"{}、 ".format(count)+  simPred, cur_socre]
                qout.put(results)
                
        return

    def crnnRecWithBox(self,im, boxes_list,score_list):
        """
        crnn模型，ocr识别
        @@model,
        @@converter,
        @@im:Array
        @@text_recs:text box
        @@ifIm:是否输出box对应的img

        """
        results = []
        boxes_list = sorted_boxes(boxes_list)

        line_imgs = []
        for index, (box, score) in enumerate(zip(boxes_list[:angle_detect_num], score_list[:angle_detect_num])):
            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            print("partImg_array.shape = {}".format(partImg_array.shape))
            partImg = Image.fromarray(partImg_array).convert("RGB")
            line_imgs.append(partImg)

        angle_res = False
        if angle_detect:
            angle_res = self.angle_handle.predict_rbgs(line_imgs)
        print("========================angle_res = {}".format(angle_res))

        count = 1
        for index, (box ,score) in enumerate(zip(boxes_list,score_list)):

            tmp_box = copy.deepcopy(box)
            partImg_array = get_rotate_crop_image(im, tmp_box.astype(np.float32))
            print("partImg_array.shape = {}".format(partImg_array.shape))

            partImg = Image.fromarray(partImg_array).convert("RGB")

            if angle_detect and angle_res:
                partImg = partImg.rotate(180)


            if not is_rgb:
                partImg = partImg.convert('L')

            try:
                if is_rgb:
                    simPred = self.crnn_handle.predict_rbg(partImg)  ##识别的文本
                else:
                    simPred = self.crnn_handle.predict(partImg)  ##识别的文本
            except Exception as e:
                print(traceback.format_exc())
                continue

            if simPred.strip() != '':
                results.append([tmp_box,"{}、 ".format(count)+  simPred,score])
                count += 1

        return results

    def text_predict(self,img,short_size):
        if run_mode == "multiprocessing":
            print("multiprocessing!!!!!!")
            self.queue_input.put(np.asarray(img).astype(np.uint8))
            result = self.qoutput.get()
        else:
            print(" one processer!!!!!!")
            boxes_list, score_list = self.text_handle.process(np.asarray(img).astype(np.uint8),short_size=short_size)
            result = self.crnnRecWithBox(np.array(img), boxes_list,score_list)

        return result


if __name__ == "__main__":
    pass
