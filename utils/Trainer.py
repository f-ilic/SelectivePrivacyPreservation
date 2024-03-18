import torch
import time
from datetime import datetime
from utils.metrics import accuracy
from utils.AverageMeter import AverageMeter
from utils.ConfusionMatrix import ConfusionMatrix
from torchvision.utils import make_grid, save_image


class Trainer(object):
    def __init__(self, simulation) -> None:
        self.sim = simulation

    def do(
        self,
        mode,
        model,
        dl,
        epoch,
        criterion,
        optimizer,
        writer,
        log_video=True,
        accumulate_grad_batches=1,
    ):
        if mode == "train":
            train = True
            model.train()
        elif mode in ["test", "ablate", "ablate_afdnoise", "eval"]:
            train = False
            model.eval()
        else:
            raise ValueError(
                f'mode must be either "train", "test" or "ablate" but was {mode}'
            )

        losses = AverageMeter()

        if dl.dataset.num_classes > 2:
            top_k_range = [1, 2, 3, 4, 5]
        else:
            top_k_range = [1, 2]

        topk_dict = {f"top{k}": AverageMeter() for k in top_k_range}

        confusion_matrix = ConfusionMatrix(
            n_classes=dl.dataset.num_classes, labels=list(dl.dataset.classes)
        )
        dt = datetime.now()
        print(
            f'\n[ {dt.strftime("%H:%M | %d-%b")}  | {mode.upper().rjust(5, " ")} | EPOCH {epoch:03d}] ',
            end="",
        )

        start_time = time.time()

        for step, (inputs, masks, flows, labels) in enumerate(dl):
            batchsize = labels.size(0)
            if type(inputs) is list:
                inputs = [i.cuda() for i in inputs]
            else:
                inputs = inputs.cuda()
            labels = labels.cuda()

            if log_video and step == 0 and type(inputs) is not list:
                inv_T = dl.dataset.inverse_normalise
                if len(inputs.shape) == 5:  # its a Video Tensor B C T H W
                    tmp = []
                    for batch_idx in range(inputs.shape[0]):
                        tmp.append(inv_T(inputs[batch_idx]))
                    tmp = torch.stack(tmp).permute(0, 2, 1, 3, 4).detach()
                    writer.add_video(f"{mode}/video", tmp, global_step=epoch)
                elif len(inputs.shape) == 4:  # its a regular B C H W
                    tmp = inv_T(inputs.permute(1, 0, 2, 3)).permute(1, 0, 2, 3)
                    writer.add_image(f"{mode}/video", make_grid(tmp), global_step=epoch)
            classification_outputs = model(inputs)

            loss = criterion(classification_outputs, labels)

            topk_accuracy = accuracy(
                classification_outputs.data, labels, topk=top_k_range
            )
            losses.update(loss.item(), batchsize)

            # update each element in the topk_dict entry with the current accuracy
            for idx, (k, v) in enumerate(topk_dict.items()):
                v.update(topk_accuracy[idx].item(), batchsize)

            class_preds = classification_outputs.data.argmax(1)
            confusion_matrix.update(class_preds.detach(), labels.detach())

            loss = loss / accumulate_grad_batches

            if train is True:
                loss.backward()

                if (step % accumulate_grad_batches == 0) or (step + 1 == len(dl)):
                    optimizer.step()
                    optimizer.zero_grad()

            for k, v in topk_dict.items():
                writer.add_scalar(f"{mode}_step/{k}", v.avg, step)

        seconds = time.time() - start_time
        # print(f"Epoch {epoch} took {seconds/60:.2f} minutes")

        writer.add_scalar(f"time/{mode}_epoch_seconds", seconds, epoch)
        writer.add_scalar(f"{mode}/loss", losses.avg, epoch)

        for k, v in topk_dict.items():
            writer.add_scalar(f"{mode}/{k}", v.avg, epoch)
            print(f"{k}: {v.avg:.2f}", end="\t")

        writer.add_image(
            f"{mode}/confusion_matrix",
            confusion_matrix.as_img(
                dpi=500, fontsize=2, label_angle=90, display_values=False
            ),
            epoch,
            dataformats="CHW",
        )

        # save data outside of tensorboard
        self.sim.save_data(
            topk_dict,
            subdir="topk",
            title=f"{mode}_topk_{epoch:05d}",
            overwrite=True,
        )
        self.sim.save_data(
            confusion_matrix.mat,
            subdir="confusionmatrix",
            title=f"{mode}_confusion_{epoch:05d}",
            overwrite=True,
        )

        if epoch == 0:
            self.sim.save_data(
                dl.dataset.classes,
                subdir="confusionmatrix",
                title=f"labels",
                overwrite=True,
            )

        return topk_dict["top1"].avg


class Trainer_E2SX3D(object):
    def __init__(self, simulation) -> None:
        self.sim = simulation

    def do(
        self,
        mode,
        model,
        dl,
        epoch,
        criterion,
        optimizer,
        writer,
        log_video=False,
        accumulate_grad_batches=1,
    ):
        if mode == "train":
            train = True
            model.train()
        elif mode in ["test", "ablate", "ablate_afdnoise", "eval"]:
            train = False
            model.eval()
        else:
            raise ValueError(
                f'mode must be either "train", "test" or "ablate" but was {mode}'
            )

        losses = AverageMeter()
        topk_dict = {f"top{k}": AverageMeter() for k in range(1, 6)}

        confusion_matrix = ConfusionMatrix(
            n_classes=dl.dataset.num_classes, labels=list(dl.dataset.classes)
        )
        dt = datetime.now()
        print(
            f'\n[ {dt.strftime("%H:%M | %d-%b")}  | {mode.upper().rjust(5, " ")} | EPOCH {epoch:03d}] ',
            end="",
        )
        start_time = time.time()

        for step, (input, masks, flows, labels) in enumerate(dl):
            rgbinput = input.cuda()
            flowinput = flows.cuda()

            batchsize = labels.size(0)
            labels = labels.cuda()
            classification_outputs = model(rgbinput, flowinput)

            loss = criterion(classification_outputs, labels)
            topk_accuracy = accuracy(
                classification_outputs.data, labels, topk=range(1, 6)
            )
            losses.update(loss.item(), batchsize / accumulate_grad_batches)

            # update each element in the topk_dict entry with the current accuracy
            for idx, (k, v) in enumerate(topk_dict.items()):
                v.update(topk_accuracy[idx].item(), batchsize / accumulate_grad_batches)

            class_preds = classification_outputs.data.argmax(1)
            confusion_matrix.update(class_preds.detach(), labels.detach())

            loss = loss / accumulate_grad_batches

            if train is True:
                loss.backward()

                if (step % accumulate_grad_batches == 0) or (step + 1 == len(dl)):
                    optimizer.step()
                    optimizer.zero_grad()

            for k, v in topk_dict.items():
                writer.add_scalar(f"{mode}_step/{k}", v.avg, step)

        seconds = time.time() - start_time

        writer.add_scalar(f"time/{mode}_epoch_seconds", seconds, epoch)
        writer.add_scalar(f"{mode}/loss", losses.avg, epoch)

        for k, v in topk_dict.items():
            writer.add_scalar(f"{mode}/{k}", v.avg, epoch)
            print(f"{k}: {v.avg:.2f}", end="\t")

        writer.add_image(
            f"{mode}/confusion_matrix",
            confusion_matrix.as_img(
                dpi=500, fontsize=2, label_angle=90, display_values=False
            ),
            epoch,
            dataformats="CHW",
        )

        # save data outside of tensorboard
        self.sim.save_data(topk_dict, subdir="topk", title=f"{mode}_topk_{epoch}")
        self.sim.save_data(
            confusion_matrix.mat,
            subdir="confusionmatrix",
            title=f"{mode}_confusion_{epoch}",
        )

        if epoch == 0:
            self.sim.save_data(
                dl.dataset.classes,
                subdir="confusionmatrix",
                title=f"labels",
                overwrite=True,
            )
        return topk_dict["top1"].avg
