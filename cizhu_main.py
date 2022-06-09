from tkinter import filedialog
import os
import traceback
import yaml
from imutils.paths import list_images
from six import StringIO
from detection_inference import load_ovdet_models,detmodel_iference,write_log,warm_loadnms
import flask
def logefile():
    if not os.path.exists('AI_LOG.txt'):
        print("create Log file")
        file = open('AI_LOG.txt', 'w')
        write_log("start running")
        file.close()
    elif os.path.getsize('AI_LOG.txt') > 100 * 1000000:  # 大于100M删除
        print("Log file size out 100M,remove it")
        os.remove("AI_LOG.txt")
        file = open('AI_LOG.txt', 'w')
        write_log("start running")
        file.close()
    else:
        pass
model_dic={}
config=yaml.load(open("config.yaml", encoding='utf-8'),Loader=yaml.FullLoader)
app = flask.Flask(__name__)
@app.route("/", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}
    print('请求ip地址为:',flask.request.remote_addr)
    print(flask.request.form)
    write_log(flask.request.form)
    # ensure an image was properly uploaded to our endpoint
    if flask.request.method == "POST":
        try:
            image_path = flask.request.form["input_image_path"]
            try:
                conf_thres = flask.request.form["conf_thrd"]
            except:
                conf_thres = config["conf_thresh"]
            try:
                iou_thres = flask.request.form["iou_thrd"]
            except:
                iou_thres = config["iou_thresh"]
            try:
                device_config = flask.request.form["device"]
            except:
                device_config = config["device"]
            pre_model_path=config["pre_model_path"]
            try:
                detmodel_type = flask.request.form["detmodel_type"]
            except:
                detmodel_type = config["detmodel_type"]
            try:
                cut_size=flask.request.form["cut_patch_size"]
            except:
                cut_size=config["cut_patch_size"]
            image_save_dir= os.path.dirname(image_path)
            data_save_path = os.path.join(image_save_dir,config["save_data_name"])
            try:
                save_patch_number = flask.request.form["save_patch_number"]
            except:
                save_patch_number = config["save_patch_number"]
            print("image_path:",image_path,"conf_thres:",conf_thres,"iou_thres:",iou_thres,"device_config:",device_config,
                  "save_patch_number:",save_patch_number,"data_save_path:",data_save_path)
            try:
                detection_model=model_dic[detmodel_type]
            except:
                print("模型%s未加载"%detmodel_type)
                model_dic[detmodel_type] = load_ovdet_models(device=device_config,weights=pre_model_path.format(detmodel_type))
                detection_model = model_dic[detmodel_type]
            detmodel_iference(input_im_path=image_path, image_save_path=image_save_dir,
                              perform_standard_pred=False, data_save_path=data_save_path,
                              conf_thres=float(conf_thres),
                              cut_size=int(cut_size), iou_thres=float(iou_thres),
                              detection_model=detection_model,patch_number=int(save_patch_number))
            data["success"] = True
        except Exception as e:
            errr_content = StringIO()
            traceback.print_exc(file=errr_content)
            write_log("[processing fial]: " + str(e) + " " + errr_content.getvalue())
        return flask.jsonify(data)
    else:
        return flask.jsonify(data)

if __name__ == '__main__':
    logefile()
    write_log("* Starting warm up...")
    warm_loadnms()
    if config["GUI_test_mode"]:
        write_log("* Starting GUI test service...")
        try:
            originpath = filedialog.askdirectory(title="请选择要处理的文件夹路径")
            image_save_dir = filedialog.askdirectory(title="请选择结果保存路径")
            data_save_path = os.path.join(image_save_dir, config["save_data_name"])
            conf_thres = config["conf_thresh"]
            iou_thres = config["iou_thresh"]
            device_config = config["device"]
            pre_model_path = config["pre_model_path"]
            detmodel_type = config["detmodel_type"]
            cut_size = config["cut_patch_size"]
            save_patch_number = config["save_patch_number"]
            detection_model = load_ovdet_models(device=device_config, weights=pre_model_path.format(detmodel_type))
            image_list=list(list_images(originpath))
            if len(image_list)==0:
                print("没有找到图片")
                os.system("pause")
            else:
                for image_path in image_list:
                    print("image_path:", image_path, "conf_thres:", conf_thres, "iou_thres:", iou_thres, "device_config:",
                          device_config)
                    detmodel_iference(input_im_path=image_path, image_save_path=image_save_dir,
                                      perform_standard_pred=False, data_save_path=data_save_path,
                                      conf_thres=float(conf_thres),
                                      cut_size=int(cut_size), iou_thres=float(iou_thres),
                                      detection_model=detection_model,patch_number=int(save_patch_number))
                write_log("* GUI test service finished")
                os.system("pause")
        except:
            errr_content = StringIO()
            traceback.print_exc(file=errr_content)
            write_log("[GUI test fial]: " + str(errr_content.getvalue()))
        os.system("pause")
    else:
        write_log("* Starting web service...")
        try:
            app.run(host='127.0.0.1',port=5011,threaded=False)
        except Exception as e:
            errr_content = StringIO()
            traceback.print_exc(file=errr_content)
            write_log("[processing fial]: " + str(e) + " " + errr_content.getvalue())

