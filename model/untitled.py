import torch

# GPU 사용 가능 -> True, GPU 사용 불가 -> False
def p() :
    print(torch.cuda.is_available())