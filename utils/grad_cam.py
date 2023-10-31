import os
import cv2
import torch
import timm
import h5py
import numpy as np

from matplotlib import pyplot as plt
from torchvision import transforms
from my_dataset import MyDataSetTCIR, MySubset_WS
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

def reshape_transform(tensor, height=24, width=24):
    """
    tensor: [batch_size, num_tokens, token_dim]
    """
    result = tensor.reshape(tensor.size(0),
        height, width, -1)
    # result = tensor[:, 1 :  , :].reshape(tensor.size(0),
    #     height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result

# idx = 100
# data_path = r'I:\TCIR-SPLT\TCIR-train.h5'
dir_path = 'figs'
data_path = r'I:\TCIR-SPLT\TCIR-test.h5'

model = timm.create_model('tiny_vit_21m_224.in1k', pretrained=False, num_classes=1,)
# model.load_state_dict(torch.load(r'I:\TCIR-SPLT\ckpt\tiny-vit\ckpt_9.633521409958362.pth')['net'])
model.load_state_dict(torch.load(r'I:\TCIR-SPLT\ckpt\tiny-vit\ckpt_9.004478918879393.pth')['net'])
model = model.eval()
# cam = GradCAM(model=model, target_layers=[model.head.norm], reshape_transform=reshape_transform)
cam = GradCAM(model=model, target_layers=[model.patch_embed.conv2.bn])


transform_test = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
    ]
)

multi_modal = False
test_subset = MyDataSetTCIR(data_path)
test_set = MySubset_WS(test_subset, transform=transform_test)

hf = h5py.File(data_path, 'r')
data_matrix = hf['matrix']

for idx in range(len(test_set)):
    img, wv, _, _, _ = test_set[idx]
    img = img.unsqueeze(0)
    # img = img.cuda()
    target_category = None # 可以指定一个类别，或者使用 None 表示最高概率的类别
    grayscale_cam = cam(input_tensor=img, targets=target_category)
    grayscale_cam = grayscale_cam[0, :]


    img = data_matrix[idx]
    im = img[:, :, 0]

    # im.shape # (201, 201)
    # resize image
    resized = cv2.resize(im, (224, 224), interpolation = cv2.INTER_AREA)
    # cv2.imwrite('img.jpg', resized)
    # plt.imshow(im, cmap='Greys')
    img_name = os.path.join(dir_path, f"{idx}_img.jpg")
    cam_name = os.path.join(dir_path, f"{idx}_cam.jpg") 
    
    plt.imsave(img_name, resized, cmap='Greys')

    img_color = cv2.imread(img_name)
    img_color = img_color.astype(np.float32)  / 255
    # resized = resized.astype(np.float32) / resized.max()
    # print(resized.max())

    # img_color = np.zeros((224, 224, 3), dtype=np.float32)
    # img_color[:, :, 0] = resized
    # img_color[:, :, 1] = resized
    # img_color[:, :, 2] = resized
    # plt.imshow(im, cmap='Greys')

    # 将 grad-cam 的输出叠加到原始图像上
    visualization = show_cam_on_image(img_color, grayscale_cam, image_weight=0.7)

    # cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR, visualization)
    cv2.imwrite(cam_name, visualization)