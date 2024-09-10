import timm
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import numpy as np
import cv2
from torchsummary import summary

# 시드 값 설정 함수
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)


# 시드 값 고정 (예: 44로 설정)
seed = 44
set_seed(seed)

# 모델과 체크포인트 경로 설정
model_path = "/shared/home/vclp/hyunwook/minyoung/defect-cls/ckpt_mixup/tf_efficientnetv2_m.in21k_stratnew10_20240909_182401/ep-4_acc-64.95.ckpt"
model_name = "tf_efficientnetv2_m.in21k"

# 모델 생성 (num_classes=2로 설정)
model = timm.create_model(model_name, pretrained=False, num_classes=2).to("cuda")
print(model.default_cfg["mean"])

# input_size를 먼저 정의합니다.
input_size = (model.default_cfg["input_size"][1], model.default_cfg["input_size"][2])

# summary 함수를 호출할 때 input_size를 전달합니다.
summary(model, input_size=(3, *input_size))

# 체크포인트 로드
checkpoint = torch.load(model_path)

# 모델의 최종 fc 레이어 크기를 체크포인트의 fc 레이어 크기에 맞게 설정
model.load_state_dict(checkpoint)

root_dir = "/shared/home"
version = "v2"

# 기존 절대경로에서 상대경로로 변환
testdir_defects = "{}/vclp/hyunwook/minyoung/defect-cls/BF/datas{}/test_defects".format(root_dir, version)
testdir_goods = "{}/vclp/hyunwook/minyoung/defect-cls/BF/datas{}/test_goods".format(root_dir, version)

# 입력 데이터에 대한 변환 정의
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),  # 이미지 크기 조정
    transforms.Grayscale(num_output_channels=3),  # 2채널 이미지를 3채널로 변환
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# datasets
test_data_defects = datasets.ImageFolder(testdir_defects, transform=test_transforms)
test_data_goods = datasets.ImageFolder(testdir_goods, transform=test_transforms)

# dataloader
test_data_defects = torch.utils.data.DataLoader(test_data_defects, shuffle=True, batch_size=16)
test_data_goods = torch.utils.data.DataLoader(test_data_goods, shuffle=True, batch_size=16)

def test_step(test_dl, goods=False):
    with torch.no_grad():
        cum_loss = 0
        test_correct = 0
        for x_batch, y_batch in test_dl:
            if goods:
                for i, x in enumerate(y_batch):
                    y_batch[i] = 1
            x_batch = x_batch.to("cuda")
            y_batch = y_batch.unsqueeze(1).float()  # convert target to same nn output shape
            y_batch = y_batch.to("cuda")

            # model to eval mode
            model.eval()

            yhat = model(x_batch)

            outs = torch.sigmoid(yhat)
            outs = (outs > 0.5).float()
            test_correct += (outs == y_batch).float().sum()
        return 100 * test_correct / (len(test_dl) * 32)

print("Defects acc: {}".format(test_step(test_data_defects).item()))
print("Goods acc: {}".format(test_step(test_data_goods, goods=True).item()))