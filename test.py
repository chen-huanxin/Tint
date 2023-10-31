import argparse
import math

import numpy as np
import os
import torch
import datetime
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms.functional import rotate, InterpolationMode

from utils import progress_bar
from my_dataset import MySubset_WS, MyDataSetTCIR
import math
from sklearn.metrics import mean_squared_error
from models.origin_resnet import get_resnet_ms

rmse = lambda y_true, y_pred: math.sqrt(mean_squared_error(y_true, y_pred))

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default="resnet32",
        choices=[
            "resnet32",
            "resnet50",
            "resnet101",
            "OriginRes50",
            "OriginRes34"
        ],
        help="Model to use",
    )

    # DeepTI_WS wind speed regression
    # DeepTI    wind type classification

    parser.add_argument(
        "--dataset",
        default="TCIR_WS",
        choices=["DeepTI_WS", "TCIR_WS"],
        help="dataset name",
    )

    parser.add_argument(
        "--dataset_root",
        default="/home/chenhuanxin/datasets/TCIR-SPLT",
        help="The path of dataset dir.",
    )

    parser.add_argument(
        "--training_mode",
        default="cross-entropy",
        choices=["contrastive", "cross-entropy", "focal", "focal_contrastive", "ce_contrastive"],
        help="Type of training use either a two steps contrastive then cross-entropy or \
                         just cross-entropy",
    )

    # 0.1, 0.25. 0.5 ...
    parser.add_argument(
        "--focal_alpla",
        default=None,
        type=float,
        help="On the contrastive step this will be multiplied by two.",
    )

    # 0, 0.1 , 5
    parser.add_argument(
        "--focal_gamma",
        default=0.25,
        type=float,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument(
        "--batch_size",
        default=32,
        type=int,
        help="On the contrastive step this will be multiplied by two.",
    )

    parser.add_argument(
        "--description",
        default="MultiScale",
        type=str,
        help="The description of the model",
    )

    parser.add_argument("--temperature", default=0.1, type=float, help="Constant for loss no thorough ")

    parser.add_argument("--auto-augment", default=False, type=bool)

    # focal contrasive alpha
    parser.add_argument("--alpha", default=1, type=float)

    parser.add_argument("--n_epochs_contrastive", default=75, type=int)
    parser.add_argument("--n_epochs_cross_entropy", default=100, type=int)

    # Train From Pretrained
    parser.add_argument("--lr_contrastive", default=0.01, type=float)
    parser.add_argument("--lr_cross_entropy", default=0.001, type=float)

    # parser.add_argument('--weights', default="checkpoint/2022_08_09_03_47_08cross-entropy/ckpt_9.624193373371858.pth", type=str, help='initial weights path')
    parser.add_argument('--weights', default="checkpoint/2022_08_10_02_27_44cross-entropy/ckpt_9.757252683257262.pth", type=str, help='initial weights path')

    parser.add_argument("--cosine", default=False, type=bool, help="Check this to use cosine annealing instead of ")

    parser.add_argument("--step", default=True, type=bool, help="Check this to use step")

    parser.add_argument("--lr_decay_rate", type=float, default=0.1, help="Lr decay rate when cosine is false")

    parser.add_argument(
        "--lr_decay_epochs",
        type=list,
        default=[50, 75],
        help="If cosine false at what epoch to decay lr with lr_decay_rate",
    )

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for SGD")

    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum for SGD")

    parser.add_argument("--num_workers", default=4, type=int, help="number of workers for Dataloader")

    parser.add_argument('--gpu', type=int, default=0, help='using gpu')

    parser.add_argument('--multi_modal', action="store_true", default=False)

    parser.add_argument('--smooth', action="store_true", default=False)

    parser.add_argument('--rotation_blend', action="store_true", default=False)

    parser.add_argument('--blend_num', type=int, default=6, help='num of blending')

    args = parser.parse_args()

    return args

def my_smooth(src):
    for idx in range(1, len(src) - 1):
        src[idx] = (src[idx - 1] + src[idx] + src[idx + 1]) / 3
    return src

def crop_center(matrix, crop_width):
    total_width = matrix.shape[2]
    start = total_width // 2 - crop_width // 2
    end = start + crop_width
    return matrix[:, :, start:end, start:end]

def rotation_blending(model, blending_num, images, loc_feats, args):
    sum_outputs = torch.zeros(images.size(0), 1).cuda()
    times = 0
    for angle in np.linspace(0, 360, blending_num, endpoint=False):
        rotated_image = rotate(images, angle,  InterpolationMode.BILINEAR, fill=0)
        output = model(rotated_image.to(args.device), loc_feats)
        sum_outputs += output
        times += 1

    return sum_outputs / times


def validation_any(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path):
    """

    :param epoch: int
    :param model: torch.nn.Module, Model
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module, Loss
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    # Calculate RMSE
    All_RMSE = []

    # for smoothing
    item_list = []

    with torch.no_grad():
        for batch_idx, (inputs, targets, loc_feats, time, id) in enumerate(test_loader):
            targets, loc_feats = targets.to(args.device), loc_feats.to(args.device)
            
            if args.rotation_blend:
                outputs = rotation_blending(model, args.blend_num, inputs, loc_feats, args)
            else:
                inputs = inputs.to(args.device)   
                outputs = model(inputs, loc_feats)

            outputs = torch.squeeze(outputs)
            predicted = outputs

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate RMSE
            labelsTrue = targets.data.cpu().numpy()
            predTrue = predicted.data.cpu().numpy()

            # for smoothing
            if args.smooth:
                tmp_list = []
                for i in range(targets.size(0)):
                    tmp_list.append([predTrue[i], labelsTrue[i], id[i], time[i]])
                item_list.extend(tmp_list)

            cal_rmse = rmse(labelsTrue, predTrue)
            All_RMSE.append(cal_rmse)

            progress_bar(
                batch_idx,
                len(test_loader),
                "Loss: %.3f | Acc: %.3f%% | RMSE: %.3f(%d/%d)"
                % (
                    test_loss / (batch_idx + 1),
                    100.0 * correct / total,
                    np.mean(All_RMSE),
                    correct,
                    total,
                ),
            )

    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE)

    # for smoothing
    if args.smooth:
        sorted(item_list, key=lambda x: x[3])
        sorted(item_list, key=lambda x: x[2])

        pred_sort = []
        label_sort = []
        for item in item_list:
            pred_sort.append(item[0])
            label_sort.append(item[1])

        pred_sort = my_smooth(pred_sort)
        smooth_rmse = rmse(pred_sort, label_sort)

        loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "Smooth rmse " + str(smooth_rmse) + "\n")
    else:
        loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "\n")

    loghandle.flush()

    print("[epoch {}], accuracy: {},  RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

def main():

    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

    args = parse_args()

    torch.cuda.set_device(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.device = device

    if not os.path.isdir("logs"):
        os.makedirs("logs")

    now_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    NowMode = args.training_mode

    ckpt_save_path = os.path.join(os.getcwd(), "checkpoint", now_time + NowMode)
    SavedNowPath = os.path.join(os.getcwd(), "logs", now_time + NowMode)

    os.makedirs(SavedNowPath)

    test_path = os.path.join(args.dataset_root, 'TCIR-test.h5')

    transform_test = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
            ]
        )

    test_subset = MyDataSetTCIR(test_path, args.multi_modal)
    test_set = MySubset_WS(test_subset, transform=transform_test)

    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = 1
    model = get_resnet_ms(args.model, num_classes)
    weights_dict = torch.load(args.weights, map_location="cpu")
    new_state_dict = {}

    for k, v in weights_dict['net'].items():
        if "module" in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model = model.to(args.device)
    print("Successful using pretrain-weights.")

    LogFile = os.path.join(SavedNowPath, "log.txt")
    loghandle = open(LogFile, 'w')
    loghandle.write(str(args) + "\n")

    description = args.description
    loghandle.write(description + "\n")

    writer = SummaryWriter("logs")

    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    validation_any(0, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path)
    loghandle.close()

if __name__ == "__main__":
    main()
