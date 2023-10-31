import time
import torch
import timm
from torchsummary import summary
from thop import profile
from torchstat import stat
from models.origin_resnet import get_resnet_ms

## ViT
model = timm.create_model('tiny_vit_21m_224.in1k', pretrained=False, num_classes=1,)
model = model.eval()

summary(model, (3, 224, 224))


model = model.cuda()
data = torch.randn(1, 3, 224, 224).cuda()
flops, params = profile(model, inputs=(data, ))
print(flops/1e9, params/1e6) # GFlops, MB


## CNN
model = get_resnet_ms("OriginRes34", 1)
model = model.eval()
stat(model, (3, 224, 224))
summary(model, input_size=(3,224,224))


## time cmp
model = get_resnet_ms("OriginRes34", 1)
model = model.eval()
model = model.cuda()

# model = timm.create_model('tiny_vit_21m_224.in1k', pretrained=False)
# model = model.eval()
# model = model.cuda()

T1 = time.time()
data = torch.rand((1,3,224,224)).cuda()
for i in range(100):
    model(data)

T2 = time.time()
# print('程序运行时间:%s毫秒' % ((T2 - T1)*1000))
print(f"time: {(T2 - T1)*10}")