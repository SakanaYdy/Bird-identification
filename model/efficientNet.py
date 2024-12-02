import torch
import torchvision
from torchvision import datasets, transforms
from torch import nn
from PIL import Image
import matplotlib.pyplot as plt


def apply_test_transforms(inp):
    out = transforms.functional.resize(inp, [224, 224])
    out = transforms.functional.to_tensor(out)
    out = transforms.functional.normalize(out, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return out


def predict(model, filepath, classes, show_img=False):
    # 读取本地文件
    im = Image.open(filepath)

    if show_img:
        plt.imshow(im)
        plt.show()  # 显示图片

    # 图像预处理
    im_as_tensor = apply_test_transforms(im)
    minibatch = torch.stack([im_as_tensor])

    # 如果有GPU可用，将数据移到GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    minibatch = minibatch.to(device)

    # 将模型移到相同的设备（CPU 或 GPU）
    model = model.to(device)

    # 进行模型预测
    pred = model(minibatch)

    # 获取预测结果
    _, classnum = torch.max(pred, 1)
    # print("Predicted Class Index:", classnum.item())

    # 返回预测的类名
    return classes[classnum.item()]


def get_classes(data_dir):
    all_data = datasets.ImageFolder(data_dir)
    return all_data.classes


def main(img_path):
    # 直接写死路径，删除命令行参数
    dataset_path = "../../CUB_200_2011/images/"  # 图像分类数据集路径
    model_path = "EfficientNet.pth"  # 训练好的模型路径
    show_img = False  # 是否显示图像

    # 获取数据集的类别
    classes = get_classes(dataset_path)

    # 加载模型
    model = torchvision.models.efficientnet_b0(pretrained=True)

    # 冻结所有模型参数
    for param in model.parameters():
        param.requires_grad = False

    # 替换模型的最后一层
    n_inputs = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Linear(n_inputs, 2048),
        nn.SiLU(),
        nn.Dropout(0.3),
        nn.Linear(2048, len(classes))
    )

    # 加载保存的模型权重
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # 进行预测
    result = predict(model, img_path, classes, show_img=show_img)  # 使用传入的图像路径
    return result
    # print("Predicted Class Name:", result)

def add(a,b):
    return a + b

# if __name__ == "__main__":
#     # 解析命令行参数
#     parser = argparse.ArgumentParser()
#     parser.add_argument("img_path", help="Path to the image for prediction")
#     args = parser.parse_args()
#
#     # 调用主函数，传入图片路径
#     main(args.img_path)
