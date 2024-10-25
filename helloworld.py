import torch
print(torch.__version__)

print(torch.cuda.is_available())  # 应该返回 True
print(torch.cuda.device_count())  # 应该返回你的 GPU 数量