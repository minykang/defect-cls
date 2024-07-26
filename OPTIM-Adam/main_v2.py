import timm, argparse, torch, csv, os, pytz, torchvision
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from torch.nn.modules.loss import BCEWithLogitsLoss
from itertools import zip_longest

import csv, pytz
from datetime import datetime


def save_experiment_results(args, best_epoch, best_test_accuracy, traindir, start_time):
    results = {
        'timestamp': start_time,
        'model_name': args.model_name,
        'lr': args.lr,
        'n_epochs': args.n_epochs,
        'version': args.version,
        'strat': args.strat,
        'batch_size': args.batch_size,
        'best_epoch': best_epoch,
        'best_test_accuracy': f"{best_test_accuracy:.4f}",
        'train_directory': traindir
    }

    csv_file = 'experiment_results.csv'
    file_exists = os.path.isfile(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"Results saved to {csv_file}")


def get_data_dir(version, strat, is_train=True):
    mode = "train" if is_train else "test"
    return f"./BF/datas{version}/strat_{strat}/{mode}"

def create_transforms(model):
    input_size = (model.default_cfg["input_size"][1], model.default_cfg["input_size"][2])
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=model.default_cfg["mean"],
            std=model.default_cfg["std"],
        ),
    ])

def get_loaders(traindir, testdir, transforms, batch_size):
    train_data = datasets.ImageFolder(traindir, transform=transforms)
    test_data = datasets.ImageFolder(testdir, transform=transforms)
    trainloader = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(test_data, shuffle=True, batch_size=batch_size)
    return trainloader, testloader, train_data, test_data

def make_train_step(model, optimizer, loss_fn):
    def train_step(x, y):
        yhat = model(x)
        loss = loss_fn(yhat, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        return loss, yhat
    return train_step

def log_results(loggings, model_name, strat, lr, epoch, epoch_loss, cum_loss, train_accuracy, test_accuracy, best_epoch, best_test_accuracy):
    loggings['model_name'].append(model_name)
    loggings['strat'].append(strat)
    loggings['lr'].append(lr)
    loggings['epoch'].append(epoch+1)
    loggings['train_loss'].append(epoch_loss.item())
    loggings['val_loss'].append(cum_loss.item())
    loggings['train_acc'].append(train_accuracy.item())
    loggings['test_acc'].append(test_accuracy.item())
    loggings['best_epoch'].append(best_epoch)
    loggings['best_eval_acc'].append(best_test_accuracy)

def save_model(model, save_path, epoch, test_accuracy):
    os.makedirs(save_path, exist_ok=True)
    best_model_path = f"{save_path}/ep-{epoch}_acc-{test_accuracy:.2f}.ckpt"
    torch.save(model.state_dict(), os.path.join(best_model_path))

def save_logs(save_path, loggings):
    with open(os.path.join(save_path, "logs.csv"), 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter='|')
        datas = [loggings[k] for k in loggings.keys()]
        exported_datas = zip_longest(*datas, fillvalue='')
        csvwriter.writerows(exported_datas)

def main(args):
    model_name = args.model_name
    lr = args.lr
    n_epochs = args.n_epochs
    version = args.version
    strat = args.strat
    batch_size = args.batch_size
    
    
    
    asia_timezone = pytz.timezone('Asia/Seoul')
    start_time = datetime.now(asia_timezone).strftime("%Y%m%d_%H%M%S")

    traindir = get_data_dir(version, strat, True)
    testdir = get_data_dir(version, strat, False)

    print("😤"*40)
    print(f"Train dir: {traindir}")
    print(f"Test dir: {testdir}")
    print(f"Model: {model_name}")
    print("😤"*40)

    model = timm.create_model(model_name, pretrained=True, num_classes=1)
    transforms = create_transforms(model)
    trainloader, testloader, train_data, test_data = get_loaders(traindir, testdir, transforms, batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    for params in model.parameters():
        params.requires_grad_ = True

    loss_fn = BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80], gamma=0.1)
    train_step = make_train_step(model, optimizer, loss_fn)

    best_epoch = 0
    best_test_accuracy = 0
    loggings = {
        "model_name": ["model_name"], "strat": ["strat"], "lr": ["lr"],
        "epoch": ["epoch"], "train_loss": ["train_loss"], "val_loss": ["val_loss"],
        "train_acc": ["train_acc"], "test_acc": ["test_acc"],
        "best_epoch": ["best_epoch"], "best_eval_acc": ["best_eval_acc"],
    }

    save_path = os.path.join("ckpt", f"{model_name}_strat{strat}_{start_time}")

    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_loss = 0
        train_correct = 0
        for x_batch, y_batch in trainloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.unsqueeze(1).float().to(device)

            loss, outs = train_step(x_batch, y_batch)
            epoch_loss += loss/len(trainloader)

            outs = torch.sigmoid(outs)
            outs = (outs > 0.5).float()
            train_correct += (outs == y_batch).float().sum()
        
        train_accuracy = 100 * train_correct / len(train_data)

        model.eval()
        with torch.no_grad():
            cum_loss = 0
            test_correct = 0
            for x_batch, y_batch in testloader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.unsqueeze(1).float().to(device)

                yhat = model(x_batch)
                val_loss = loss_fn(yhat, y_batch)
                cum_loss += val_loss/len(testloader)

                outs = torch.sigmoid(yhat)
                outs = (outs > 0.5).float()
                test_correct += (outs == y_batch).float().sum()
            
            test_accuracy = 100 * test_correct / len(test_data)

            if test_accuracy > best_test_accuracy:
                best_epoch = epoch
                best_test_accuracy = test_accuracy.item()
                save_model(model, save_path, epoch, test_accuracy)

            log_results(loggings, model_name, strat, lr, epoch, epoch_loss, cum_loss, 
                        train_accuracy, test_accuracy, best_epoch, best_test_accuracy)

        scheduler.step()

    save_logs(save_path, loggings)

    print(f"Best epoch: {best_epoch}")
    print(f"Best test accuracy: {best_test_accuracy:.4f}")
    print(f"Train directory: {traindir}")
    
    save_experiment_results(args, best_epoch, best_test_accuracy, traindir, start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="resnet50.tv_in1k", help="model name")
    parser.add_argument("--version", type=str, default="v3", help="version")
    parser.add_argument("--strat", type=str, default="bilstering", help="strategy")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    args = parser.parse_args()

    main(args)