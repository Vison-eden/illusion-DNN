import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data.data_loader import testDataset
from models.models import Brainmodels
from transform import transform_test
import argparse

# 设置命令行参数
parser = argparse.ArgumentParser(description='Test the trained DNN model')
parser.add_argument('--m_name', type=str, help="The model selection for testing")
parser.add_argument('--batch_size', type=int, default=32, help="Batch size for testing")
parser.add_argument('--checkpoint_path', type=str, help="Path to the saved model checkpoint")
parser.add_argument('--test_path', type=str, default="./dataset/test/", help="Path to the test dataset")
parser.add_argument('--workers', type=int, default=8, help="Number of workers for data loading")
parser.add_argument('--classes', type=int, default=2, help="Number of classes in the dataset")
args = parser.parse_args()

# 加载测试数据集
test_data = testDataset(data_dir=args.test_path, transform=transform_test)
test_loader = DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Brainmodels(args.m_name, args.classes, pretrained=False).final_models()
checkpoint = torch.load(args.checkpoint_path)
model.load_state_dict(checkpoint)
model = model.to(device)
model.eval()  # 设置模型为评估模式

# 定义损失函数
criterion = nn.CrossEntropyLoss().to(device)

# 测试过程
test_loss = 0.0
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# 打印测试结果
test_loss /= len(test_loader)
test_accuracy = 100 * correct / total
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')

