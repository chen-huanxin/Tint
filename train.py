import argparse
import math
import shutil
import numpy as np
import os
import torch
import datetime
import shutil
import timm

from torch.backends import cudnn
from torch import nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from utils import progress_bar
from data_augmentation.auto_augment import AutoAugment
from models.origin_resnet import get_resnet_ms
from my_dataset import MyDataSetDeepTI, MyDataSetTCIR, MySubset_WS
from sklearn.metrics import mean_squared_error

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
            "OriginRes34",
            "Vit",
        ],
        help="Model to use",
    )

    parser.add_argument(
        "--vit_name",
        default="vit_base_patch16_224",
        choices=[
            "vit_base_patch16_224",
            "tiny_vit_21m_224.in1k",
            "fastvit_t8.apple_dist_in1k",
        ],
        help="Model to use",
    )

    parser.add_argument(
        "--dataset",
        default="TCIR_WS",
        choices=["DeepTI_WS", "TCIR_WS"],
        help="dataset name",
    )

    parser.add_argument(
        "--dataset_root",
        default=r"I:\TCIR-SPLT",
        # default="/home/chenhuanxin/datasets/TCIR-SPLT",
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
        "--img_size",
        default=224,
        type=int,
        help="Image size",
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

    #parser.add_argument('--weights', type=str,
    #                    default='/media/dell/564C2A944C2A6F45/LinuxCode/TyphoonEstimation/Pretrained/resnet50-19c8e357.pth',
    #                    help='initial weights path')

    parser.add_argument('--weights', default=r"I:\Models\resnet34-333f7ec4.pth", type=str, help='initial weights path')

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

    parser.add_argument("--num_workers", default=0, type=int, help="number of workers for Dataloader")

    parser.add_argument('--gpu', type=int, default=0, help='using gpu')

    parser.add_argument('--multi_gpu', action="store_true", default=False)

    parser.add_argument('--use_test', action="store_true", default=False)

    parser.add_argument('--multi_modal', action="store_true", default=False)

    args = parser.parse_args()

    return args

def adjust_learning_rate(optimizer, epoch, loghandle, mode, args):
    """

    :param optimizer: torch.optim
    :param epoch: int
    :param mode: str
    :param args: argparse.Namespace
    :return: None
    """
    if mode == "contrastive":
        lr = args.lr_contrastive
        n_epochs = args.n_epochs_contrastive
    elif mode == "cross_entropy":
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy
    else:
        lr = args.lr_cross_entropy
        n_epochs = args.n_epochs_cross_entropy

    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / n_epochs)) / 2

    if args.step:
        n_steps_passed = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if n_steps_passed > 0:
            lr = lr * (args.lr_decay_rate ** n_steps_passed)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    loghandle.write("Adjusting Lr = " + str(lr) + "\n")
    loghandle.flush()


def train_any(model, train_loader, test_loader, criterion, optimizer, writer, args, loghandle, ckpt_save_path):
    """

    :param model: torch.nn.Module Model
    :param train_loader: torch.utils.data.DataLoader
    :param test_loader: torch.utils.data.DataLoader
    :param criterion: torch.nn.Module Loss
    :param optimizer: torch.optim
    :param writer: torch.utils.tensorboard.SummaryWriter
    :param args: argparse.Namespace
    :return:
    """

    criterion = torch.nn.MSELoss()

    for epoch in range(args.n_epochs_cross_entropy):  # loop over the dataset multiple times
        print("Epoch [%d/%d]" % (epoch + 1, args.n_epochs_cross_entropy))

        model.train()
        train_loss = 0
        correct = 0
        total = 0
        if args.dataset == "TCIR_WS":
            for batch_idx, (inputs, targets, _, _, _) in enumerate(train_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                optimizer.zero_grad()

                if args.model == "GoogleNet":
                    outputs = model(inputs).logits
                else:
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)

                targets = targets.float()

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs

                total_batch = targets.size(0)
                correct_batch = predicted.eq(targets).sum().item()
                total += total_batch
                correct += correct_batch

                writer.add_scalar(
                    "Loss train | Cross Entropy",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

                writer.add_scalar(
                    "Accuracy train | Cross Entropy",
                    correct_batch / total_batch,
                    epoch * len(train_loader) + batch_idx,
                )

                progress_bar(
                    batch_idx,
                    len(train_loader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        else:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(args.device), targets.to(args.device)

                optimizer.zero_grad()

                if args.model == "GoogleNet":
                    outputs = model(inputs).logits
                else:
                    outputs = model(inputs)
                    outputs = torch.squeeze(outputs)

                targets = targets.float()

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = outputs

                total_batch = targets.size(0)
                correct_batch = predicted.eq(targets).sum().item()
                total += total_batch
                correct += correct_batch

                writer.add_scalar(
                    "Loss train | Cross Entropy",
                    loss.item(),
                    epoch * len(train_loader) + batch_idx,
                )

                writer.add_scalar(
                    "Accuracy train | Cross Entropy",
                    correct_batch / total_batch,
                    epoch * len(train_loader) + batch_idx,
                )

                progress_bar(
                    batch_idx,
                    len(train_loader),
                    "Loss: %.3f | Acc: %.3f%% (%d/%d)"
                    % (
                        train_loss / (batch_idx + 1),
                        100.0 * correct / total,
                        correct,
                        total,
                    ),
                )

        loghandle.write("CE Training Epoch = " + str(epoch) +
        " Loss train = " + str(loss.item()) + "Accuracy train = " + str(correct_batch / total_batch) + "\n")
        loghandle.flush()

        validation_any(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path)

        adjust_learning_rate(optimizer, epoch, loghandle, mode='cross_entropy', args=args)

    print("Finished Training")

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

    with torch.no_grad():
        for batch_idx, (inputs, targets, _, _, _) in enumerate(test_loader):
            inputs, targets = inputs.to(args.device), targets.to(args.device)

            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            predicted = outputs

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate RMSE
            labelsTrue = targets.data.cpu().numpy()
            predTrue = predicted.data.cpu().numpy()

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

    # Save checkpoint.
    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE)
    loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "\n")
    loghandle.write("Best Accuracy: " + str(args.best_acc) + "Best RMSE: " + str(args.best_rmse) +"\n")

    loghandle.flush()

    print("[epoch {}] , accuracy: {}, RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if myrmse < args.best_rmse:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "rmse": myrmse,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_rmse = myrmse

        Name = "ckpt_" + str(args.best_rmse) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_acc = acc

        Name = "ckpt_" + str(args.best_acc) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

def validation(epoch, model, test_loader, criterion, writer, args, loghandle, ckpt_save_path):
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

    with torch.no_grad():
        for batch_idx, (inputs, targets, loc_feats) in enumerate(test_loader):
            inputs, targets, loc_feats = inputs.to(args.device), targets.to(args.device), loc_feats.to(args.device)

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Calculate RMSE
            labelsTrue = targets.data.cpu().numpy()
            predTrue = predicted.data.cpu().numpy()

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

    # Save checkpoint.
    acc = 100.0 * correct / total
    myrmse = np.mean(All_RMSE)
    loghandle.write("Epoch" + str(epoch) + "Tesing Accuracy " + str(acc) + "Tesing rmse " + str(myrmse) + "\n")
    loghandle.write("Best Accuracy: " + str(args.best_acc) + "Best RMSE: " + str(args.best_rmse) +"\n")
    loghandle.flush()

    print("[epoch {}] , accuracy: {}, RMSE: {}".format(epoch, acc, myrmse))

    writer.add_scalar("Accuracy validation | Cross Entropy", acc, epoch)

    if myrmse < args.best_rmse:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "rmse": myrmse,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_rmse = myrmse

        Name = "ckpt_" + str(args.best_rmse) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

    if acc > args.best_acc:
        print("Saving..")
        state = {
            "net": model.state_dict(),
            "acc": acc,
            "epoch": epoch,
        }
        if not os.path.isdir("checkpoint"):
            os.mkdir("checkpoint")

        if not os.path.isdir(ckpt_save_path):
            os.makedirs(ckpt_save_path)

        args.best_acc = acc

        Name = "ckpt_" + str(args.best_acc) + ".pth"
        savepath = os.path.join(ckpt_save_path, Name)
        torch.save(state, savepath)

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

    sourcefile = "train.py"
    destfile = os.path.join(SavedNowPath, sourcefile)
    shutil.copyfile(sourcefile, destfile)

    sourcefile = "my_dataset.py"
    destfile = os.path.join(SavedNowPath, sourcefile)
    shutil.copyfile(sourcefile, destfile)

    sourcefile = "models/origin_resnet.py"
    destfile = os.path.join(SavedNowPath, "origin_resnet.py")
    shutil.copyfile(sourcefile, destfile)

    if args.dataset == "DeepTI_WS":
        transform_train = [
            # transforms.Resize(224),
            transforms.Resize(args.img_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]
        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                transforms.ToTensor(),
            ]
        )

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
            ]
        )

        train_set = MyDataSetDeepTI("Train", transform=transform_train)
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        test_set = MyDataSetDeepTI("Val", transform=transform_test)
        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

        num_classes = 1

    if args.dataset == "TCIR_WS":
        train_path = os.path.join(args.dataset_root, 'TCIR-train.h5')
        if args.use_test:
            test_path = os.path.join(args.dataset_root, 'TCIR-test.h5')
        else:
            test_path = os.path.join(args.dataset_root, 'TCIR-val.h5')

        transform_train = [
            # transforms.Resize(224),
            transforms.Resize(args.img_size),
            transforms.RandomRotation(20),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip()
        ]

        if args.auto_augment:
            transform_train.append(AutoAugment())

        transform_train.extend(
            [
                transforms.ToTensor(),
            ]
        )

        transform_train = transforms.Compose(transform_train)

        transform_test = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
            ]
        )

        train_subset = MyDataSetTCIR(train_path, args.multi_modal)
        train_set = MySubset_WS(train_subset, transform=transform_train)

        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
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

    if args.model == 'Vit':
        # model_name = 'vit_base_patch16_224'  # 选择您需要的模型名称
        # model_name = 'tiny_vit_21m_224.in1k'  # 选择您需要的模型名称
        model_name = args.vit_name
        print(model_name)

        model = timm.create_model(model_name, pretrained=True, num_classes=1,)

        # # 修改模型的输出层
        # num_classes = 1  # 例如，设置为 10 个类别
        # model.head = nn.Linear(model.head.in_features, num_classes)

        print("Use Vision Transformer.")
        # ## 加载自己的预训练模型
        # if args.weights is not None:
        #     model.load_state_dict(torch.load(args.weights)['net'])
        #     print("Successful using myself pretrain-weights.")
        print("Successful using pretrain-weights.")
    else:
        model = get_resnet_ms(args.model, num_classes)

        # Adding Pretrain model
        if os.path.exists(args.weights):
            weights_dict = torch.load(args.weights, map_location="cpu")
            new_state_dict = {}
            for k, v in weights_dict.items():
                # Transfer TC classification into Regression
                if "linear" in k :
                    continue

                if "layer" in k:

                    NewLayerCaption_1 = k.replace("layer", "layer1_")
                    NewLayerCaption_2 = k.replace("layer", "layer2_")
                    NewLayerCaption_3 = k.replace("layer", "layer3_")

                    new_state_dict[NewLayerCaption_1] = v
                    new_state_dict[NewLayerCaption_2] = v
                    new_state_dict[NewLayerCaption_3] = v

                    continue

                if "bn1" in k :
                    NewLayerCaption_1 = k.replace("bn1", "bn2")
                    NewLayerCaption_2 = k.replace("bn1", "bn3")

                    new_state_dict[k] = v
                    new_state_dict[NewLayerCaption_1] = v
                    new_state_dict[NewLayerCaption_2] = v

                    continue

                if "conv1" in k:
                    NewLayerCaption_1 = k.replace("conv1", "conv2")
                    NewLayerCaption_2 = k.replace("conv1", "conv3")

                    new_state_dict[k] = v
                    new_state_dict[NewLayerCaption_1] = v
                    new_state_dict[NewLayerCaption_2] = v

            model.load_state_dict(new_state_dict, strict=False)

            print("Successful using pretrain-weights.")
        else:
            print("not using pretrain-weights.")

    if torch.cuda.device_count() > 1 and args.multi_gpu:

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model, device_ids=[0,1,2,3])

    model = model.to(args.device)

    cudnn.benchmark = True

    LogFile = os.path.join(SavedNowPath, "log.txt")
    file_handle = open(LogFile, 'w')
    file_handle.write(str(args) + "\n")

    description = args.description
    file_handle.write(description + "\n")

    writer = SummaryWriter("logs")
    if args.model == "Vit":
        # args.lr_cross_entropy = 0.00001
        optimizer = optim.AdamW(model.parameters(),
            lr=args.lr_cross_entropy,
            # momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr_cross_entropy,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )


    criterion = nn.CrossEntropyLoss()
    criterion.to(args.device)

    args.best_acc = 0.0
    args.best_rmse = 100.0
    train_any(model, train_loader, test_loader, criterion, optimizer, writer, args, file_handle,
                         ckpt_save_path)

    file_handle.close()


if __name__ == "__main__":
    main()
