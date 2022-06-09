import cv2
import numpy as np
import os
import sys
import time
import traceback
from lsnms import nms
from pathlib import Path
from sahi.slicing import slice_image
from six import StringIO
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())
from utils.general import scale_coords, letterbox, \
    plot_cricle, \
    convert_from_image_to_cv2, convert_from_cv2_to_image, use_lnms_nms
from openvino.inference_engine import IECore
def write_log(data):
    with open("Ai_LOG.txt", "a") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + " " * 5 + str(data) + "\n")

def get_openvino_core_net_exec(model_xml_path, target_device="CPU"):
    if not os.path.isfile(model_xml_path):
        print(f'{model_xml_path} does not exist')
        write_log(f'{model_xml_path} does not exist')
        return None
    model_bin_path = Path(model_xml_path).with_suffix('.bin').as_posix()
    # load IECore object
    OVIE = IECore()
    # load openVINO network
    OVNet = OVIE.read_network(model=model_xml_path, weights=model_bin_path)
    # create executable network
    if target_device!="CPU":
        write_log("openvino Using {} inference".format(target_device))
    else:
        target_device="CPU"
        write_log("openvino Using CPU inference")
    OVExec = OVIE.load_network( network=OVNet, device_name=target_device)
    return OVIE, OVNet, OVExec

def get_prediction(image,
    detection_model,
    image_size: int = None,
    shift_amount=None,
    conf_thres: float = 0.5,  # confidence threshold
    iou_thres: float = 0.45,  # NMS IOU threshold
    full_shape=None):
    if shift_amount is None:
        shift_amount = [0, 0]
    dets = []
    for i in range(len(image)):
        img = letterbox(convert_from_image_to_cv2(image[i]), detection_model["model_input_size"], stride=64, auto=False)[0]
        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = img.astype('float32')
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        results = detection_model["OVExec"].infer(inputs={detection_model["InputLayer"]: img})
        detections = results[detection_model["OutputLayer"]]
        det = use_lnms_nms(detections, conf_thres=conf_thres, iou_thres=iou_thres, cutoff_distance=8)
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image[i].shape).round()
            det[:, 0] = det[:, 0] + shift_amount[i][0]
            det[:, 1] = det[:, 1] + shift_amount[i][1]
            det[:, 2] = det[:, 2] + shift_amount[i][0]
            det[:, 3] = det[:, 3] + shift_amount[i][1]
            dets.append(det)
    return dets

def load_ovdet_models(weights,device):
    OVIE, OVNet, OVExec = get_openvino_core_net_exec(weights, device)
    InputLayer = next(iter(OVNet.input_info))
    OutputLayer = list(OVNet.outputs)[-1]
    print("Available Devices: ", OVIE.available_devices)
    # print("Input Layer: ", InputLayer)
    # print("Output Layer: ", OutputLayer)
    print(weights.split("/")[-2]," Model Input Shape: ", OVNet.input_info[InputLayer].input_data.shape)
    print(weights.split("/")[-2]," Model Output Shape: ", OVNet.outputs[OutputLayer].shape)
    imgsz = max(OVNet.input_info[InputLayer].input_data.shape)
    return {
        "OVExec": OVExec,
        "InputLayer": InputLayer,
        "OutputLayer": OutputLayer,
        "model_input_size": imgsz,
    }

def warm_loadnms():
    s_time=time.time()
    # Create boxes: approx 30 pixels wide / high
    image_size = 1000
    n_predictions = 1000
    topleft = np.random.uniform(0.0, high=image_size, size=(n_predictions, 2))
    wh = np.random.uniform(10, 25, size=topleft.shape)
    boxes = np.concatenate([topleft, topleft + wh], axis=1).astype(np.float64)
    # Create scores
    scores = np.random.uniform(0., 1., size=(len(boxes),))
    # Apply NMS
    # During the process, overlapping boxes are queried using a R-Tree, ensuring a log-time search
    keep = nms(boxes, scores, iou_threshold=0.5)
    print("warm load nms cost %.2f s"%(time.time()-s_time))

def detmodel_iference(
    input_im_path,
    image_save_path,
    data_save_path,
    cut_size: int = 640,
    overlap_height_ratio: float = 0.1,
    overlap_width_ratio: float = 0.1,
    perform_standard_pred: bool = True,
    conf_thres: float = 0.2,  # confidence threshold
    iou_thres: float = 0.1,  # NMS IOU threshold
    cutoff_distance: int = 64,
    detection_model=None,
    distance_tred=0,
    patch_number=1,
    line_thickness=1):
    Label = ['IMB', 'IMBA']
    custom_colors = [(0, 255, 0), (0, 0, 255)]
    durations_in_seconds = {}
    time_start = time.time()
    im0 = cv2.imdecode(np.fromfile(input_im_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    # perform sliced prediction
    pred_time = time.time()
    object_prediction_list = []
    if perform_standard_pred:
        # perform standard prediction
        prediction_result = get_prediction(
            image=[im0],
            detection_model=detection_model,
            image_size=cut_size,
            shift_amount=[0, 0],
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            full_shape=None,
        )
        object_prediction_list.extend(prediction_result)
        object_prediction_list = [object_prediction for object_prediction in object_prediction_list if
                                  isinstance(object_prediction, np.ndarray)]
    else:
        write_log("开始切割图片")
        slice_image_result = slice_image(
            image=convert_from_cv2_to_image(im0),
            slice_height=cut_size,
            slice_width=cut_size,
            overlap_height_ratio=overlap_height_ratio,
            overlap_width_ratio=overlap_width_ratio,
        )
        del im0
        write_log("图片切割完成")
        num_slices = len(slice_image_result)
        durations_in_seconds["slice_number"]=num_slices
        durations_in_seconds["slice_time"] = time.time() - time_start
        num_batch=num_slices
        num_group = int(num_slices / num_batch)
        print("Number of slices:", num_slices,"Number Batches:",num_batch)
        for group_ind in range(num_group):
            # prepare batch (currently supports only 1 batch)
            image_list = []
            shift_amount_list = []
            for image_ind in range(num_batch):
                image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
                shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
            # perform batch prediction
            write_log("开始批量预测")
            prediction_result = get_prediction(
                image=image_list,
                detection_model=detection_model,
                image_size=cut_size,
                shift_amount=shift_amount_list,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ]
            )
            object_prediction_list.extend(prediction_result)
        object_prediction_list = [object_prediction for object_prediction in object_prediction_list if
                                  isinstance(object_prediction, np.ndarray)]
    write_log("批量预测完成")
    anysis_result_title = '细胞编号' + "," +'X坐标' + "," + 'Y坐标' +"," + '直径'+"\n"
    anysis_result = ""
    split_result=[]
    write_result_and_plot_start = time.time()
    im0 = cv2.imdecode(np.fromfile(input_im_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    write_log("开始写入结果")
    if object_prediction_list:
        csplit_num=1
        im0 = cv2.imdecode(np.fromfile(input_im_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        split_height = im0.shape[0] // patch_number
        for xx in range(0,im0.shape[0]-(im0.shape[0] % split_height),split_height):
            cv2.imencode('.jpg', im0[xx:xx + split_height, :])[1].tofile(os.path.join(image_save_path, r"Chamber1_%i_BR.jpg" % (csplit_num)))
            csplit_num+=1
        det = np.concatenate(object_prediction_list, axis=0)
        durations_in_seconds["prediction"] = time.time() - pred_time
        nms_start = time.time()
        keep = nms(det[:, :4], det[:, 4], iou_threshold=iou_thres, score_threshold=conf_thres,
                   cutoff_distance=cutoff_distance)
        det = det[keep]
        print("Number of Cell:", det.shape[0])
        # WBweighted box clustering
        # parrl_xyxy, label, cluster_indices = wbc(det[:,:4].cpu().numpy().astype(np.float64), det[:,4].cpu().numpy().astype(np.float64), iou_threshold=iou_thres, score_threshold=conf_thres,
        #                                                    cutoff_distance=cutoff_distance)
        durations_in_seconds["NMS"] = time.time() - nms_start
        array_xyxy = det[:, :4]
        cell_info = np.empty([len(det), 13])
        cell_info[:, 0] = ((array_xyxy[:, 0] + array_xyxy[:, 2]) / 2).astype(int)  # center x
        cell_info[:, 1] = ((array_xyxy[:, 1] + array_xyxy[:, 3]) / 2).astype(int)  # center y
        cell_info[:, 2] = ((array_xyxy[:, 2] - array_xyxy[:, 0]) + (array_xyxy[:, 3] - array_xyxy[:, 1])) / 4  # 半径
        cell_info[:, 3] = np.ones(len(det))  # 团属性
        cell_info[:, 4] = det[:, -1].astype(int)  # 标签
        cell_info[:, 5] = ((array_xyxy[:, 2] - array_xyxy[:, 0]) + (array_xyxy[:, 3] - array_xyxy[:, 1])) / 2  # 直径
        cell_info[:, 6] = array_xyxy[:, 0]  # tx
        cell_info[:, 7] = array_xyxy[:, 1]  # ty
        cell_info[:, 8] = array_xyxy[:, 2]  # bx
        cell_info[:, 9] = array_xyxy[:, 3]  # by
        cell_info[:, 10] = np.multiply((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                       (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 面积
        cell_info[:, 11] = np.maximum((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                      (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 长轴
        cell_info[:, 12] = np.minimum((array_xyxy[:, 2] - array_xyxy[:, 0]),
                                      (array_xyxy[:, 3] - array_xyxy[:, 1]))  # 短轴
        write_result_and_plot_start = time.time()
        IMB_area_list=[]
        IMB_diam_list = []
        first_add=0
        for cell_num, (cellx, celly, redius, group_type, label_num, diam, xyxy_0, xyxy_1, xyxy_2, xyxy_3, Area, max_aix,
                  min_aix) in enumerate(cell_info):
            if label_num==0:
                IMB_area_list.append(Area)
                IMB_diam_list.append(diam)
                split_result.append([int(cellx),int(celly),diam])
                anysis_result += str(cell_num+first_add + 1)+","+",".join([str(i) for i in [int(cellx),int(celly),diam]]) + "\n"
                plot_cricle((int(cellx), int(celly)), redius, im0, color=custom_colors[int(label_num)],line_thickness=line_thickness)
            else:
                area_means=np.mean(np.array(IMB_area_list)) if len(IMB_area_list)>10 else 100
                diam_means = np.mean(np.array(IMB_diam_list)) if len(IMB_diam_list) > 0 else 10
                split_number=Area//area_means
                for split_cell in range(int(split_number)):
                    split_result.append([int(cellx), int(celly),"%.1f"%(diam/split_number)])
                    anysis_result += str(cell_num+first_add + 1) + "," + ",".join([str(i) for i in [int(cellx), int(celly), "%.1f"%(diam_means)]]) + "\n"
                    plot_cricle((int(cellx), int(celly)), redius, im0, color=custom_colors[int(label_num)],
                                line_thickness=line_thickness)
                    first_add += 1
    try:
        if os.path.exists(data_save_path):
            os.remove(data_save_path)
        with open(data_save_path, 'a+', encoding='utf-8') as f:
            f.write(anysis_result_title + anysis_result)
    except Exception as e:
        errr_content = StringIO()
        traceback.print_exc(file=errr_content)
        write_log("[processing fial]: " + str(e) + " " + errr_content.getvalue())
    try:
        cv2.imencode('.jpg', im0)[1].tofile(os.path.join(image_save_path,"result_light.jpg"))
        split_height=im0.shape[0]//patch_number
        Ty = 0
        spltnum=0
        for xx in range(0,im0.shape[0]-(im0.shape[0] % split_height),split_height):
            signtl_anysis_result=""
            By=Ty+split_height
            if len(split_result)>0:
                cell_num = 1
                for cell_item in split_result:
                    if (cell_item[1]<=By) and (cell_item[1]>=Ty):
                        signtl_anysis_result += str(cell_num) + "," + ",".join([str(i) for i in cell_item]) + "\n"
                        cell_num += 1
                        split_result=list(filter(lambda x:x!=cell_item,split_result))
            Ty+=split_height
            cv2.imencode('.jpg', im0[xx:xx + split_height, :])[1].tofile(os.path.join(image_save_path,r"Chamber1_%i_BR_light.jpg" % (spltnum + 1)))
            if os.path.exists(os.path.join(image_save_path, r"Chamber1_%i_BR_light_status.txt" % (spltnum + 1))):
                os.remove(os.path.join(image_save_path, r"Chamber1_%i_BR_light_status.txt" % (spltnum + 1)))
            with open(os.path.join(image_save_path, r"Chamber1_%i_BR_light_status.txt" % (spltnum + 1)), 'a+', encoding='utf-8') as f:
                f.write(anysis_result_title + signtl_anysis_result)
            spltnum+=1
    except Exception as e:
        errr_content = StringIO()
        traceback.print_exc(file=errr_content)
        write_log("[processing fial]: " + str(e) + " " + errr_content.getvalue())
    write_log("写入结果完成")
    durations_in_seconds["write_result_and_plot"] = time.time() - write_result_and_plot_start
    print(durations_in_seconds)
    write_log(durations_in_seconds)
    print("总处理时间", time.time() - time_start)