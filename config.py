import os



filt_path = os.path.abspath(__file__)
father_path = os.path.abspath(os.path.dirname(filt_path) + os.path.sep + ".")

# dbnet 参数
dbnet_model_path = os.path.join(father_path, "models/dbnet.onnx")
dbnet_max_size = 6000 #长边最大长度
short_size = 960
pad_size = 0 #检测是pad尺寸，有些文档文字充满整个屏幕检测有误，需要pad


# crnn参数
box_standard_height = 32
crnn_lite = True
is_rgb = True
crnn_model_path = os.path.join(father_path, "models/crnn_lite_lstm.onnx")
crnn_width_step = 300



# angle
angle_detect_weight = 192
angle_detect = True
angle_detect_num = 30
angle_net_path = os.path.join(father_path, "models/angle_net.onnx")


max_post_time = 100 # ip 访问最大次数

from crnn.keys import alphabetChinese as alphabet


white_ips = [] #白名单

version = 'api/v1'

run_mode = "multiprocessing"
