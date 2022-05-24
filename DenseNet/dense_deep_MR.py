import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
#import torchvision.models as models
from torch.autograd import Variable


def conv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(in_dim, out_dim, kernel_size=1, stride=1, padding=0),
                         nn.AvgPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
                         nn.ELU(True),
                         nn.UpsamplingNearest2d(scale_factor=2))


def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
    else:
        block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
    if bn:
        block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s.relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))
    if bn:
        block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class G2(nn.Module):
    def __init__(self, input_nc, output_nc, nf):
        super(G2, self).__init__()
        # input is 256 x 256
        layer_idx = 1
        name = 'layer%d' % layer_idx
        layer1 = nn.Sequential()
        layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))
        # input is 128 x 128
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer2 = blockUNet(nf, nf * 2, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 64 x 64
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer3 = blockUNet(nf * 2, nf * 4, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 32
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer4 = blockUNet(nf * 4, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 16
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer5 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 8
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer6 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 4
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer7 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)
        # input is 2 x  2
        layer_idx += 1
        name = 'layer%d' % layer_idx
        layer8 = blockUNet(nf * 8, nf * 8, name, transposed=False, bn=True, relu=False, dropout=False)

        ## NOTE: decoder
        # input is 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8
        dlayer8 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=False, relu=True, dropout=True)

        # import pdb; pdb.set_trace()
        # input is 2
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer7 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 4
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer6 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=True)
        # input is 8
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer5 = blockUNet(d_inc, nf * 8, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 16
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 8 * 2
        dlayer4 = blockUNet(d_inc, nf * 4, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 32
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 4 * 2
        dlayer3 = blockUNet(d_inc, nf * 2, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 64
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        d_inc = nf * 2 * 2
        dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)
        # input is 128
        layer_idx -= 1
        name = 'dlayer%d' % layer_idx
        dlayer1 = nn.Sequential()
        d_inc = nf * 2
        dlayer1.add_module('%s.relu' % name, nn.ReLU(inplace=True))
        dlayer1.add_module('%s.tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
        dlayer1.add_module('%s.tanh' % name, nn.LeakyReLU(0.2, inplace=True))

        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.layer4 = layer4
        self.layer5 = layer5
        self.layer6 = layer6
        self.layer7 = layer7
        self.layer8 = layer8
        self.dlayer8 = dlayer8
        self.dlayer7 = dlayer7
        self.dlayer6 = dlayer6
        self.dlayer5 = dlayer5
        self.dlayer4 = dlayer4
        self.dlayer3 = dlayer3
        self.dlayer2 = dlayer2
        self.dlayer1 = dlayer1

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        out7 = self.layer7(out6)
        out8 = self.layer8(out7)
        dout8 = self.dlayer8(out8)
        dout8_out7 = torch.cat([dout8, out7], 1)
        dout7 = self.dlayer7(dout8_out7)
        dout7_out6 = torch.cat([dout7, out6], 1)
        dout6 = self.dlayer6(dout7_out6)
        dout6_out5 = torch.cat([dout6, out5], 1)
        dout5 = self.dlayer5(dout6_out5)
        dout5_out4 = torch.cat([dout5, out4], 1)
        dout4 = self.dlayer4(dout5_out4)
        dout4_out3 = torch.cat([dout4, out3], 1)
        dout3 = self.dlayer3(dout4_out3)
        dout3_out2 = torch.cat([dout3, out2], 1)
        dout2 = self.dlayer2(dout3_out2)
        dout2_out1 = torch.cat([dout2, out1], 1)
        dout1 = self.dlayer1(dout2_out1)
        return dout1


class BottleneckEncoderBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(BottleneckEncoderBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2*32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3*32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4*32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5*32)
        self.relu6= nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.conv6 = nn.Conv2d(in_planes + 5*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)      
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        if self.droprate > 0:
            out = F.dropout(out6, p=self.droprate, inplace=False, training=self.training)
        #out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out6, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([out5, out6], 1)


class BottleneckDecoderBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckDecoderBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(in_planes + 32)
        self.relu2 = nn.ReLU(inplace=True)
        self.bn3 = nn.BatchNorm2d(in_planes + 2*32)
        self.relu3 = nn.ReLU(inplace=True)
        self.bn4 = nn.BatchNorm2d(in_planes + 3*32)
        self.relu4 = nn.ReLU(inplace=True)
        self.bn5 = nn.BatchNorm2d(in_planes + 4*32)
        self.relu5 = nn.ReLU(inplace=True)
        self.bn6 = nn.BatchNorm2d(in_planes + 5*32)
        self.relu6= nn.ReLU(inplace=True)
        self.bn7 = nn.BatchNorm2d(inter_planes)
        self.relu7= nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes + 32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes + 2*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_planes + 3*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.conv5 = nn.Conv2d(in_planes + 4*32, 32, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.conv6 = nn.Conv2d(in_planes + 5*32, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.conv7 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)        
        self.droprate = dropRate

    def forward(self, x):
        out1 = self.conv1(self.relu1(self.bn1(x)))
        out1 = torch.cat([x, out1], 1)
        out2 = self.conv2(self.relu2(self.bn2(out1)))
        out2 = torch.cat([out1, out2], 1)
        out3 = self.conv3(self.relu3(self.bn3(out2)))
        out3 = torch.cat([out2, out3], 1)
        out4 = self.conv4(self.relu4(self.bn4(out3)))
        out4 = torch.cat([out3, out4], 1)
        out5 = self.conv5(self.relu5(self.bn5(out4)))
        out5 = torch.cat([out4, out5], 1)
        out6 = self.conv6(self.relu6(self.bn6(out5)))
        out = self.conv7(self.relu7(self.bn7(out6)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        #out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)


class ResidualBlock(nn.Module):
    def __init__(self, in_planes, dropRate=0.0):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)					   
        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.conv2(x1)
        out = x + x2
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return out


class TransitionBlockEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return self.pool1(out)


class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)


class Dense_decoder(nn.Module):
    def __init__(self, num_out):
        super(Dense_decoder, self).__init__()
        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        self.residual_block53 = ResidualBlock(128)
        self.residual_block54 = ResidualBlock(128)
        self.residual_block55 = ResidualBlock(128)
        self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        self.residual_block63 = ResidualBlock(64)
        self.residual_block64 = ResidualBlock(64)
        self.residual_block65 = ResidualBlock(64)
        self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        self.residual_block73 = ResidualBlock(32)
        self.residual_block74 = ResidualBlock(32)
        self.residual_block75 = ResidualBlock(32)
        self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 32)
        self.trans_block8 = TransitionBlock(64, 16)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        self.residual_block83 = ResidualBlock(16)
        self.residual_block84 = ResidualBlock(16)
        self.residual_block85 = ResidualBlock(16)
        self.residual_block86 = ResidualBlock(16)
        self.conv_out = nn.Conv2d(32, num_out, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4, opt):
        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        #x5 = self.residual_block53(x5)
        #x5 = self.residual_block54(x5)
        #x5 = self.residual_block55(x5)
        #x5 = self.residual_block56(x5)
        x52 = torch.cat([x5, x1], 1)
        #print(x52.shape)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        #x6 = self.residual_block63(x6)
        #x6 = self.residual_block64(x6)
        #x6 = self.residual_block65(x6)
        #x6 = self.residual_block66(x6)		
        #print(x6.shape)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        #x7 = self.residual_block73(x7)
        #x7 = self.residual_block74(x7)
        #x7 = self.residual_block75(x7)
        #x7 = self.residual_block76(x7)
        #print(x7.shape)
        ##  128 X 128
        #x8 = self.trans_block8(self.dense_block8(x7))
        #x8 = self.residual_block81(x8)
        #x8 = self.residual_block82(x8)
        #print(x8.shape)
        #x8 = self.residual_block83(x8)
        #x8 = self.residual_block84(x8)
        #x8 = self.residual_block85(x8)
        #x8 = self.residual_block86(x8)
        out = self.conv_out(x7)
        
        return out


class Dense_decoder2(nn.Module):
    def __init__(self, num_out):
        super(Dense_decoder2, self).__init__()
        ############# Block5-up  16-16 ##############
        self.dense_block5 = BottleneckDecoderBlock(384, 256)
        self.trans_block5 = TransitionBlock(640, 128)
        self.residual_block51 = ResidualBlock(128)
        self.residual_block52 = ResidualBlock(128)
        self.residual_block53 = ResidualBlock(128)
        self.residual_block54 = ResidualBlock(128)
        self.residual_block55 = ResidualBlock(128)
        self.residual_block56 = ResidualBlock(128)

        ############# Block6-up 32-32   ##############
        self.dense_block6 = BottleneckDecoderBlock(256, 128)
        self.trans_block6 = TransitionBlock(384, 64)
        self.residual_block61 = ResidualBlock(64)
        self.residual_block62 = ResidualBlock(64)
        self.residual_block63 = ResidualBlock(64)
        self.residual_block64 = ResidualBlock(64)
        self.residual_block65 = ResidualBlock(64)
        self.residual_block66 = ResidualBlock(64)

        ############# Block7-up 64-64   ##############
        self.dense_block7 = BottleneckDecoderBlock(64, 64)
        self.trans_block7 = TransitionBlock(128, 32)
        self.residual_block71 = ResidualBlock(32)
        self.residual_block72 = ResidualBlock(32)
        self.residual_block73 = ResidualBlock(32)
        self.residual_block74 = ResidualBlock(32)
        self.residual_block75 = ResidualBlock(32)
        self.residual_block76 = ResidualBlock(32)
        ## 128 X  128
        ############# Block8-up c  ##############
        self.dense_block8 = BottleneckDecoderBlock(32, 16)
        self.trans_block8 = TransitionBlock(32, 32)
        self.residual_block81 = ResidualBlock(16)
        self.residual_block82 = ResidualBlock(16)
        self.residual_block83 = ResidualBlock(16)
        self.residual_block84 = ResidualBlock(16)
        self.residual_block85 = ResidualBlock(16)
        self.residual_block86 = ResidualBlock(16)
        self.conv_out = nn.Conv2d(32, num_out, kernel_size=3, stride=1, padding=1)
        self.upsample = F.upsample_nearest
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, x1, x2, x4, opt):
        x42 = torch.cat([x4, x2], 1)
        ## 16 X 16
        x5 = self.trans_block5(self.dense_block5(x42))
        x5 = self.residual_block51(x5)
        x5 = self.residual_block52(x5)
        # x5 = self.residual_block53(x5)
        # x5 = self.residual_block54(x5)
        # x5 = self.residual_block55(x5)
        # x5 = self.residual_block56(x5)
        x52 = torch.cat([x5, x1], 1)
        # print(x52.shape)
        ##  32 X 32
        x6 = self.trans_block6(self.dense_block6(x52))
        x6 = self.residual_block61(x6)
        x6 = self.residual_block62(x6)
        # x6 = self.residual_block63(x6)
        # x6 = self.residual_block64(x6)
        # x6 = self.residual_block65(x6)
        # x6 = self.residual_block66(x6)
        # print(x6.shape)
        ##  64 X 64
        x7 = self.trans_block7(self.dense_block7(x6))
        x7 = self.residual_block71(x7)
        x7 = self.residual_block72(x7)
        x8 = self.trans_block8(x7)
        # x7 = self.residual_block73(x7)
        # x7 = self.residual_block74(x7)
        # x7 = self.residual_block75(x7)
        # x7 = self.residual_block76(x7)
        # print(x7.shape)
        ##  128 X 128
        # x8 = self.trans_block8(self.dense_block8(x7))
        # x8 = self.residual_block81(x8)
        # x8 = self.residual_block82(x8)
        # print(x8.shape)
        # x8 = self.residual_block83(x8)
        # x8 = self.residual_block84(x8)
        # x8 = self.residual_block85(x8)
        # x8 = self.residual_block86(x8)
        out = self.conv_out(x8)

        return out


class Dense(nn.Module):
    def __init__(self, num_out):
        super(Dense, self).__init__()
        ############# First downsampling  ##############
        self.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool0 = nn.MaxPool2d(2,stride=2)
        ############# First Dense Layer   ##############
        self.dense11 = BottleneckEncoderBlock(64)
        self.dense1 = TransitionBlockEncoder(256,128)
       
        ############# Second Dense Layer  ##############
        self.dense21 = BottleneckEncoderBlock(128)
        self.dense22 = BottleneckEncoderBlock(320)
        self.dense2 = TransitionBlockEncoder(512,256)

        ############# Third Dense Layer  ##############
        self.dense31 = BottleneckEncoderBlock(256)
        self.dense32 = BottleneckEncoderBlock(448)
        self.dense33 = BottleneckEncoderBlock(640)
        self.dense34 = BottleneckEncoderBlock(832)
        self.dense3 = TransitionBlockEncoder(1024,512)
        

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckDecoderBlock(512, 256)#512
        self.trans_block4 = TransitionBlock(768, 128)#768
        self.residual_block41 = ResidualBlock(128)
        self.residual_block42 = ResidualBlock(128)

        self.decoder_out = Dense_decoder(num_out=num_out)
        

    def forward(self, x, opt):
        ## 256x256
        x0 = self.pool0(self.conv0(x))

        ## 64 X 64
        x1 = self.dense1(self.dense11(x0))
        # print x1.size()
        ###  32x32
        x2 = self.dense2(self.dense22(self.dense21(x1)))
        # print  x2.size()

        ### 16 X 16
        x3 = self.dense3(self.dense34(self.dense33(self.dense32(self.dense31(x2)))))

        # x3=Variable(x3.data,requires_grad=True)

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))
		
        x4 = self.residual_block41(x4)
        x4 = self.residual_block42(x4)
        

        ######################################
        out = self.decoder_out(x, x1, x2, x4, opt)               

        return out


class Dense512(nn.Module):
    def __init__(self, num_out):
        super(Dense512, self).__init__()
        ############# First downsampling  ##############
        self.conv0 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool0 = nn.MaxPool2d(2, stride=2)
        self.dense0 = TransitionBlockEncoder(64, 64)
        ############# First Dense Layer   ##############
        self.dense11 = BottleneckEncoderBlock(64)
        self.dense1 = TransitionBlockEncoder(256, 128)

        ############# Second Dense Layer  ##############
        self.dense21 = BottleneckEncoderBlock(128)
        self.dense22 = BottleneckEncoderBlock(320)
        self.dense2 = TransitionBlockEncoder(512, 256)

        ############# Third Dense Layer  ##############
        self.dense31 = BottleneckEncoderBlock(256)
        self.dense32 = BottleneckEncoderBlock(448)
        self.dense33 = BottleneckEncoderBlock(640)
        self.dense34 = BottleneckEncoderBlock(832)
        self.dense3 = TransitionBlockEncoder(1024, 512)

        ############# Block4-up  8-8  ##############
        self.dense_block4 = BottleneckDecoderBlock(512, 256)  # 512
        self.trans_block4 = TransitionBlock(768, 128)  # 768
        self.residual_block41 = ResidualBlock(128)
        self.residual_block42 = ResidualBlock(128)

        self.decoder_out = Dense_decoder2(num_out=num_out)
        ## final upsampling ###

    def forward(self, x, opt):
        ## 256x256
        x0 = self.pool0(self.conv0(x))
        x0 = self.dense0(x0)
        ## 64 X 64
        x10 = self.dense11(x0)
        x1 = self.dense1(x10)
        # print x1.size()
        ###  32x32
        x2 = self.dense2(self.dense22(self.dense21(x1)))
        # print  x2.size()

        ### 16 X 16
        x3 = self.dense3(self.dense34(self.dense33(self.dense32(self.dense31(x2)))))

        # x3=Variable(x3.data,requires_grad=True)

        ## 8 X 8
        x4 = self.trans_block4(self.dense_block4(x3))

        x4 = self.residual_block41(x4)
        x4 = self.residual_block42(x4)

        ######################################
        out = self.decoder_out(x, x1, x2, x4, opt)


        return out
