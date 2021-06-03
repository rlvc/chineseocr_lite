from PIL import  Image
import numpy as np
import cv2
import TopsInference
import onnxruntime as rt
import time
import os

class AngleNetHandle:
    def __init__(self, model_path,size_h = 32, size_w = 192):

        self.sess = rt.InferenceSession(model_path)
        self.engine_name_template = model_path.replace('.onnx', '') + \
            ('_bs{{}}_h{{}}_w{{}}_{}.exec').\
            format(TopsInference.__version__.replace(' ', ''))
        self.engine_name = ""
        self.model_path = model_path
        print("Create AngleNet enflame binary template {}".format(self.engine_name_template))
        self.handle = TopsInference.set_device(0, 0)
        self.engine = TopsInference.PyEngine()
        self.size_h = size_h
        self.size_w = size_w

    def predict_rbg(self, im):
        """
        预测
        """
        scale = im.size[1] * 1.0 / self.size_h
        w = im.size[0] / scale
        w = int(w)
        img = im.resize((w, self.size_h), Image.BILINEAR)

        if w < self.size_w:
            imgnew = Image.new('RGB', (self.size_w, self.size_h), (255))
            imgnew.paste(img, (0, 0, w, self.size_h))
        else :
            imgnew = img.crop((0, 0, self.size_w,   self.size_h))

        img = np.array(imgnew, dtype=np.float32)

        img -= 127.5
        img /= 127.5
        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)

        print("========================AngleNetHandle transformed_image.shape = {}".format(transformed_image.shape))
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
            onnx_parser.set_input_names("input")
            onnx_parser.set_output_names("out")
            module = onnx_parser.read(self.model_path)
            optimizer = TopsInference.create_optimizer()
            print("build engine ...")
            self.engine = optimizer.build(module)
            print("build engine finished.")
            self.engine.save_executable(engine_model_name)
            self.engine_name = engine_model_name
            print("save engine file: \'{}\'".format(engine_model_name))

        preds = []
        self.engine.run([transformed_image.astype(np.float32, order='C')], preds,
                       TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)

        pred = np.argmax(preds[0])

        return pred

    def predict_rbgs(self, imgs):
        nlen = len(imgs)
        res_sum = sum([self.predict_rbg(im) for im in imgs])
        return  res_sum < nlen//2


if __name__ == "__main__":
    crnn_handle = AngleNetHandle(model_path="../models/angle_net.onnx")
    import glob
    imgs = glob.glob("/Users/yanghuiyu/Desktop/myself/OCR/mbv3_crnn/test_imgs/*p*g")
    for im_path in imgs:
        im = Image.open(im_path).convert("RGB")

        print(im_path , crnn_handle.predict_rbg(im))