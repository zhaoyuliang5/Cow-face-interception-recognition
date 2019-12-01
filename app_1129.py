import faster.tools.demo_1128 as demo
import cv2
import config
import util
import os
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    sess, net = demo.restore_model('60000')
    util.set_img_format()
    config.model = 'xception_face'
    # get the keras original model object
    keras_module_face = util.get_model_class_instance()
    # load the best trained model
    keras_model_face = keras_module_face.load()
    # get the classes dict
    classes_cows_face = util.get_classes_in_keras_format()
    video_cap = cv2.VideoCapture("D:/2019-11-28pm/192.168.1.31_01_20191128151533146.mp4")
    num = 0
    while True:
        ret_val, img = video_cap.read()
        num = num + 1
        if num == 30:
            num = 0
            img = cv2.resize(img, (1920, 1080))
            img = cv2.copyMakeBorder(img, 100, 100, 200, 200, cv2.BORDER_CONSTANT, value=[0, 0, 0])
            out_img, all_coordinates_face = demo.demo(sess, net, img)
            top3_dic_names_face = {}
            if len(all_coordinates_face) != 0:
                margin = 10
                for arry in all_coordinates_face:
                    arry[-4] -= margin
                    arry[-3] -= margin
                    arry[-2] += margin
                    arry[-1] += margin
                all_coordinates_face, top3_dic_names_face = demo.keras_id_predict(img, all_coordinates_face, keras_model_face,
                                                                                  classes_cows_face)
                for box in all_coordinates_face:
                    cv2.putText(out_img, str(top3_dic_names_face[box[0]][0:6]), (box[1], box[2]-40), cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
                    pass
            print(all_coordinates_face,top3_dic_names_face)
            out_img = cv2.resize(out_img, (1440,720))
            cv2.imshow("show", out_img)
            cv2.waitKey(1)
