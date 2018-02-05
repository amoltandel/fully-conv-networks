import torch
from torch import nn
from torchvision.models import vgg16
import torch.nn.functional as F
from torch.autograd import Variable

class VGGFCN16(nn.Module):
    def __init__(self, pretrained=False, num_classes=11):
        super(VGGFCN16, self).__init__()
        self.num_classes = num_classes
        self.layers = list(vgg16(pretrained).features)

        self.output1 = nn.Conv2d(512, 4096, (1, 1), stride=7)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(p = 0.5, inplace=True)
        self.output2 = nn.Conv2d(4096, 4096, (1, 1), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(p = 0.5, inplace=True)

        self.convscale7 = nn.Conv2d(4096, num_classes, (1, 1), stride=(1, 1), padding=(1, 1))
        self.convscale4 = nn.Conv2d(512, num_classes, (1, 1), stride=(1, 1), padding=(1, 1))
        self.convscale3 = nn.Conv2d(256, num_classes, (1, 1), stride=(1, 1), padding=(1, 1))
        self.model_name = 'vgg16'
        self.final_scale = nn.Sequential(
                nn.ConvTranspose2d(num_classes, num_classes, 2, stride=(2, 2)),
                nn.ConvTranspose2d(num_classes, num_classes, 2, stride=(2, 2)),
                nn.ConvTranspose2d(num_classes, num_classes, 2, stride=(2, 2)),
            )

    def forward(self, inp):
        output = inp
        input_size = inp.size()[2:]

        for i in range(17):
            output = self.layers[i](output)
        output1 = output

        for i in range(17, 24):
            output = self.layers[i](output)
        output2 = output

        for i in range(24, 31):
            output = self.layers[i](output)
        output = self.output1(output)
        self.relu1(output)
        self.dropout1(output)
        output = self.output2(output)
        self.relu2(output)
        self.dropout2(output)
        output3 = output

        output3 = self.convscale7(output3)
        output2 = self.convscale4(output2)
        output1 = self.convscale3(output1)

        result_size = output1.size()[2:]
        output3 = F.upsample(output3, size=result_size, mode='bilinear')
        output2 = F.upsample(output2, size=result_size, mode='bilinear')

        final_output = output3 + output2 + output1
        final_output = self.final_scale(final_output)
        final_output = F.upsample(final_output, size=input_size, mode='bilinear')

        return final_output

if __name__ == '__main__':
    a = VGGFCN16()
    b = Variable(torch.FloatTensor(1, 3, 1536, 2048))
    c = a(b)
    print(b.size())
    print(c.size())
