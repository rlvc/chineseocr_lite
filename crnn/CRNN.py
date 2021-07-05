from PIL import  Image
import numpy as np
import cv2
from .keys import alphabetChinese as alphabet
# from keys import alphabetChinese as alphabet

import onnxruntime as rt
# from util import strLabelConverter, resizeNormalize
from .util import strLabelConverter, resizeNormalize
import TopsInference
import os

converter = strLabelConverter(''.join(alphabet))

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


class CRNNHandle:
    def __init__(self, model_path):

        self.sess = rt.InferenceSession(model_path)
        self.engine_name_template = model_path.replace('.onnx', '') + \
            ('_bs{{}}_h{{}}_w{{}}_{}.exec').\
            format(TopsInference.__version__.replace(' ', ''))
        self.engine_name = ""
        self.model_path = model_path
        print("Create CRNN enflame binary template {}".format(self.engine_name_template))
        self.handle = TopsInference.set_device(0, 0)
        self.engine = TopsInference.PyEngine()

    def predict(self, image):
        """
        预测
        """
        print("========================CRNNHandle image.size = {}".format(image.size))
        scale = image.size[1] * 1.0 / 32
        w = image.size[0] / scale
        w = int(w)
        transformer = resizeNormalize((w, 32))

        image = transformer(image)

        image = image.transpose(2, 0, 1)
        print("========================CRNNHandle timage.size = {}".format(image.size))
        transformed_image = np.expand_dims(image, axis=0)

        # preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})
        preds = []
        self.engine.run([transformed_image.astype(np.float32, order='C')], preds,
                       TopsInference.TIF_ENGINE_RSC_IN_HOST_OUT_HOST)
        preds = preds[0]


        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)


        sim_pred = converter.decode(preds, length, raw=False)

        return sim_pred



    def predict_rbg(self, im):
        """
        预测
        """
        print("im.size = {}".format(im.size))
        scale = im.size[1] * 1.0 / 32
        w = im.size[0] / scale
        w = int(w)
        imgnew = Image.new('RGB', (1000, 32), (255))

        img = im.resize((w, 32), Image.BILINEAR)
        imgnew.paste(img, (0, 0, w, 32))
        print("rim.size = {}".format(img.size))
        img = np.array(imgnew, dtype=np.float32)
        print("img.shape = {}".format(img.shape))
        img -= 127.5
        img /= 127.5
        image = img.transpose(2, 0, 1)
        transformed_image = np.expand_dims(image, axis=0)
        print("========================CRNNHandle transformed_image.shape = {}".format(transformed_image.shape))
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
        # preds = self.sess.run(["out"], {"input": transformed_image.astype(np.float32)})

        preds = preds[0]


        length  = preds.shape[0]
        preds = preds.reshape(length,-1)

        # preds = softmax(preds)


        preds = np.argmax(preds,axis=1)

        preds = preds.reshape(-1)

        sim_pred = converter.decode(preds, length, raw=False)

        return sim_pred



if __name__ == "__main__":
    im = Image.open("471594277244_.pic.jpg")
    crnn_handle = CRNNHandle(model_path="../models/crnn_lite_lstm_bk.onnx")
    print(crnn_handle.predict(im))