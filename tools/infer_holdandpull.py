import pickle
import cv2
import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from datetime import datetime
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from tangle.model import SepPositionNet, SepDirectionNet

if __name__ == "__main__":
    # model

    root_dir = "C:\\Users\\xinyi\\Documents"
    # model_ckpt = os.path.join(root_dir, "Checkpoints", "try_SC", "model_epoch_30.pth")
    model_ckpt = os.path.join(root_dir, "Checkpoints", "try_SR_", "model_epoch_1.pth")
    
    # density_model = DensityClassifier(img_height=IMG_HEIGHT, img_width=IMG_WIDTH)
    model = SepDirectionNet(in_channels=5).cuda()
    model.load_state_dict(torch.load(model_ckpt))

    # cuda
    use_cuda = torch.cuda.is_available()
#    use_cuda = False
    if use_cuda:
        torch.cuda.set_device(0)
        density_model = density_model.cuda()

    prediction = Prediction(density_model, IMG_HEIGHT, IMG_WIDTH, use_cuda)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset_dir = os.path.join(root_dir, "Dataset", "HoldAndPullDirectionData")
    dataset_dir = os.path.join(root_dir, "Dataset", "HoldAndPullDirectionData_Test")
    from tangle import PoseDataset
    from tangle.utils import random_inds
    inds = random_inds(100,10000)
    # train_dataset = DragKptDataset(512,512,img_folder, kpt_folder, inds)
    # train_dataset = MaskDataset(512, 512, data_folder, inds)
    # train_dataset = PullDataset(512, 512, data_folder, inds)
    # test_dataset = PoseDataset(512, 512, dataset_dir, inds)

    # test_dataset = DensityDataset('data/%s/images'%dataset_dir,
    #                        'data/%s/annots'%dataset_dir, IMG_HEIGHT, IMG_WIDTH, transform)

    itvl = 16
    from tangle.utils import *
    img = cv2.imread("D:\\datasets\\holdandpull_test\\depth22.png")
    depth = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
    drawn = depth.copy()
    grasps = [] # pull_x, pull_y, hold-x, hold_y
    # =================== click if needed ====================
    def on_click(event,x,y,flags,param):
        global mouseX,mouseY
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(drawn,(x,y),5,(0,255,0),-1)
            grasps.append([x,y])
            mouseX,mouseY = x,y
    cv2.namedWindow('click: pull and hold')
    cv2.setMouseCallback('click: pull and hold',on_click)
    while(len(grasps)<2):
        cv2.imshow('click: pull and hold',drawn)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k==ord('q'):
            break
    cv2.destroyAllWindows()
    print(grasps)
    # ========================================================

    pull_p = grasps[0]
    hold_p = grasps[1]
    # grasps = np.array([pull_p, hold_p])
    scores = []
    for r in range(itvl):
        img = transform(depth).cuda()
        heatmap = gauss_2d_batch(IMG_HEIGHT, IMG_WIDTH, 8, grasps).cuda()
        img_t = torch.cat((img, heatmap), 0)

        rot_degree = r*(360/itvl)
        direction = direction2vector(rot_degree)
        direction =  torch.from_numpy(direction).cuda()

        img_t = img_t.view(-1, img_t.shape[0], img_t.shape[1], img_t.shape[2])
        dir_t = direction.view(-1, direction.shape[0])
        
        label = density_model.forward(img_t.float(), dir_t.float())
        label = torch.nn.Softmax(dim=1)(label)
        label = label.detach().cpu().numpy()
        # print(img_t.shape, dir_t.shape, "==>", label.shape)
        scores.append(label.ravel()[1]) # only success possibility
        #print("degree: ", rot_degree, " success? : ", np.round(label, 2))\
    
    img = cv2.imread("D:\\datasets\\holdandpull_test\\depth22.png")
    from tangle.utils import draw_vectors_bundle
    drawn = draw_vectors_bundle(img=depth.copy(), start_p=pull_p, scores=scores)
    drawn = cv2.circle(drawn, pull_p,5,(0,255,0),-1)
    drawn = cv2.circle(drawn, hold_p,5,(0,255,0),-1)
    print(np.round(scores, 3))

    # cv2.imshow("windows", drawn)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()    
    plt.imshow(cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB))
    # plt.imshow(visualize_tensor(heatmap[1]), alpha=0.5)
    # plt.imshow(visualize_tensor(heatmap[0]), alpha=0.3)
    plt.show()


#     acc = 0

#     for i, data in enumerate(test_dataset):
#         vis = data[0]
#         data = [Variable(d.cuda() if use_cuda else d) for d in data]
#         img, direction, lbl_gt = data

#         img = img.view(-1, img.shape[0], img.shape[1], img.shape[2])
#         direction = direction.view(-1, direction.shape[0])

#         label = density_model.forward(img.float(), direction.float())[0]

#         label = label.detach().cpu().numpy()
#         lbl_gt = lbl_gt.detach().cpu().numpy()
#         # plt.imshow(vis[0], cmap='gray')
#         # plt.imshow(vis[3], alpha=0.5)
#         # print("pred: ", label, "| gt: ", lbl_gt)
#         # print("pred: ", label.argmax(), "| gt: ", lbl_gt.argmax())

#         if  label.argmax() == lbl_gt.argmax(): acc+= 1
#         # prediction.plot(vis, label, image_id=i)
# #
#         # plt.show()
#     print(f"accuracy: {acc}/100")
