from facenet_pytorch import MTCNN, InceptionResnetV1
import glob
import cv2
import copy
from face_grading import *
import datetime
import os
import imageio

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

mtcnn_s = MTCNN(thresholds=[0.6, 0.7, 0.7])
mtcnn = MTCNN(thresholds=[0.6, 0.7, 0.7], keep_all=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
resnet_gpu = InceptionResnetV1(pretrained='vggface2', device='cuda').eval()
font = cv2.FONT_HERSHEY_SIMPLEX

Fix_Size = (200, 200)
EPS = 24
ALPHA = 0.3


def compute_iou(rec1, rec2):
    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])
    # computing the sum_area
    sum_area = S_rec1 + S_rec2
    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])
    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def image_process_batch(boxes_lst, img_Protect, X_ADV):
    R_radio = 0.3
    Area_lst = []
    shape_lst = []
    location_lst = []
    for boxes in boxes_lst:
        w = (boxes[3] - boxes[1]) * R_radio
        h = (boxes[2] - boxes[0]) * R_radio

        A = max(int(boxes[0] - h), 1), \
            max(int(boxes[1] - w), 1), \
            min(int(boxes[2] + h), np.shape(img_Protect)[1] - 1), \
            min(int(boxes[3] + w), np.shape(img_Protect)[0] - 1)

        AREA = img_Protect[A[1]: A[3], A[0]: A[2]]
        Ori_shape = np.shape(AREA)[:2][::-1]

        AREA = Image.fromarray(AREA)
        AREA = AREA.resize(Fix_Size)

        Area_lst.append(np.array(AREA))
        shape_lst.append(Ori_shape)
        location_lst.append(A)

    Area_lst = np.array(Area_lst, dtype=np.float32)
    tensor_AREA = torch.tensor(Area_lst).cuda().requires_grad_()

    with torch.enable_grad():
        loss = -resnet_gpu(tensor_AREA + X_ADV).norm()
    grad = torch.autograd.grad(loss, [X_ADV])[0]
    X_ADV = ALPHA * X_ADV.detach() + (1 - ALPHA) * torch.sign(grad.detach()) * (EPS / 2)
    X_ADV = torch.clamp(X_ADV, -EPS, EPS)
    print(loss)

    for i in range(len(location_lst)):
        AREA = Area_lst[i]
        Ori_shape = shape_lst[i]
        A = location_lst[i]

        ADV_AREA = Image.fromarray(np.uint8(np.clip(AREA + X_ADV.cpu().detach().numpy()[0], 0, 255)))
        ADV_AREA = ADV_AREA.resize(Ori_shape)
        ADV_AREA = np.array(ADV_AREA)
        img_Protect[A[1]: A[3], A[0]: A[2]] = ADV_AREA
    return img_Protect


def P_demo(img):
    img_plot = copy.copy(img)
    img_Protect = copy.copy(img)
    batch_boxes, batch_probs = mtcnn.detect(img, landmarks=False)
    if batch_boxes is not None:
        for boxes in batch_boxes:
            img_plot = cv2.rectangle(np.asarray(img_plot), (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])), (255, 2, 0), 2)
        img_Protect = image_process_batch(batch_boxes, img_Protect, X_ADV)
    return np.uint8(img_plot), np.uint8(img_Protect)


def camera():
    cap = cv2.VideoCapture(0)
    while True:
        try:
            begin_time = datetime.datetime.now()
            ret, frame = cap.read()
            img = np.asarray(frame[..., ::-1])

            img_plot, img_Protect = P_demo(img)
            cv2.imshow('out', img_Protect[..., ::-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = datetime.datetime.now()
            during_time = end_time - begin_time
            print(during_time)
        except:
            pass
    cap.release()


def video(path):
    cap = cv2.VideoCapture(path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out_cap = cv2.VideoWriter("out.mp4", fourcc, fps, (frame_width, frame_height))
    out_cap = imageio.get_writer("out.mp4", fps=fps)
    while True:
        try:
            begin_time = datetime.datetime.now()
            ret, frame = cap.read()
            img = np.asarray(frame[..., ::-1])

            img_plot, img_Protect = P_demo(img)
            cv2.imshow('out', img_Protect[..., ::-1])
            # out_cap.write(img_Protect[..., ::-1])
            out_cap.append_data(img_Protect)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            end_time = datetime.datetime.now()
            during_time = end_time - begin_time
            print(during_time)
        except:
            break
    cap.release()
    # out_cap.release()
    out_cap.close()


if __name__ == '__main__':
    X_ADV = torch.zeros([1, 200, 200, 3]).cuda()
    X_ADV.requires_grad_()
    # camera()
    video("film.mp4")
