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
import torchvision.models as models
from torch.autograd import Variable


def _bn_function_factory(norm, relu, conv):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = conv(relu(norm(concated_features)))
        return bottleneck_output

    return bn_function


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseCBAMLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseCBAMLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('cbam', CBAM(bn_size * growth_rate))
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),

        self.drop_rate = drop_rate
        self.memory_efficient = memory_efficient

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.norm1, self.relu1, self.conv1)
        bottleneck_output = bn_function(*prev_features)
        new_features = self.conv2(self.relu2(self.cbam(self.norm2(bottleneck_output))))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        #self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
        self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != 'trans':
                new_features = layer(*features)
                features.append(new_features)
        #se = self.se(torch.cat(features, 1))
        return self.trans(torch.cat(features, 1))


class _DenseCBAMBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, num_output_features, bn_size=4, growth_rate=32, drop_rate=0, memory_efficient=False):
        super(_DenseCBAMBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseCBAMLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)
        #self.add_module('se', _SE(num_input_features + num_layers* growth_rate, num_input_features + num_layers* growth_rate//16))
        self.add_module('trans', nn.Conv2d(num_input_features + num_layers* growth_rate, num_output_features, kernel_size=(1,1), stride=(1,1)))
    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            if name != 'trans':
                new_features = layer(*features)
                features.append(new_features)
        #se = self.se(torch.cat(features, 1))
        return self.trans(torch.cat(features, 1))


class TransitionBlockEncoder(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlockEncoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=(1,1), stride=(1,1),
                                        padding=(0,0), bias=False)
        self.pool1 = nn.MaxPool2d(2,stride=2)
        self.droprate = dropRate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return self.pool1(out)


class TransitionBlockDecoder(nn.Module):
    def __init__(self, in_planes, out_planes, cubic = False, dropRate=0.0):
        super(TransitionBlockDecoder, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                                        padding=0, bias=False)
        self.droprate = dropRate
        self.cubic = cubic

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        if self.cubic:
            return F.upsample_bilinear(out, scale_factor=2)
        else:
            return F.upsample_nearest(out, scale_factor=2)


class Encoder(nn.Module):
    def __init__(self, pretrain = True, inter_planes=128, out_planes=256, block_config=(4, 4), growth_rate =32):
        super(Encoder, self).__init__()
        ############# Encoder 0 - 256 ##############
        self.conv0 = models.densenet121(pretrained = pretrain).features.conv0
        self.pool0 = models.densenet121().features.pool0
        self.dense1 = models.densenet121(pretrained = pretrain).features.denseblock1
        self.trans1 = models.densenet121(pretrained= pretrain).features.transition1
        self.dense2 = _DenseBlock(num_layers=block_config[0], num_input_features=128, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans2 = TransitionBlockEncoder(inter_planes, inter_planes)
        ############# Encoder 3 - 32 ##########################
        self.dense3 = _DenseCBAMBlock(num_layers=block_config[1], num_input_features=inter_planes, num_output_features=inter_planes,
                                  growth_rate=growth_rate)
        self.trans3 = TransitionBlockEncoder(inter_planes, inter_planes)
        self.dense4 = _DenseCBAMBlock(num_layers=block_config[2], num_input_features=inter_planes, num_output_features=out_planes,
                                  growth_rate=growth_rate)
        ############# Decoder 0 -32 ##############################
    def forward(self, x):
        out0 = self.dense1(self.pool0(self.conv0(x)))
        out1 = self.dense2(self.trans1(out0))
        out2 = self.dense3(self.trans2(out1))
        return out0, out1, out2


class Decoder(nn.Module):
    def __init__(self, in_planes=64, inter_planes = 128, out_planes = 32, block_config=(4, 4, 4), growth_rate = 32):
        super(Decoder, self).__init__()
        ############# Decoder 0 - 256 ##############
        self.TransDecoder0 = TransitionBlockDecoder(in_planes, inter_planes)
        ############# Decoder 1 - 128 ########################
        num_feat = inter_planes
        self.DenseDecoder0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_feat + 256,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[0] * growth_rate)//2)
        num_feat = (num_feat + block_config[0] * growth_rate)//2
        self.TransDecoder1 = TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 2 - 64  ########################
        self.DenseDecoder1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_feat + 256,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[1] * growth_rate)//2)
        num_feat = (num_feat + block_config[1] * growth_rate)//2
        self.TransDecoder2= TransitionBlockDecoder(num_feat, num_feat)
        ############# Decoder 3 - 32 ##########################
        self.DenseDecoder2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_feat,
                                         growth_rate=growth_rate, num_output_features=(num_feat + block_config[2] * growth_rate)//2)
        num_feat = (num_feat + block_config[2] * growth_rate)//2
        ############# Final  ##############################
        self.TransDecoder3 = TransitionBlockDecoder(num_feat, out_planes, cubic=True)
    def forward(self, x0, x1, x2):
        '''
        :param x0: 256 x 128 x 128
        :param x1: 512 x 64 x 64
        :param x2: 512 x 32 x 32
        :return:
        '''
        out3 = self.TransDecoder0(x2)
        out3 = torch.cat([x1, out3], 1)
        out4 = self.TransDecoder1(self.DenseDecoder0(out3))
        out4 = torch.cat([x0, out4], 1)
        out5 = self.TransDecoder2(self.DenseDecoder1(out4))
        out = self.TransDecoder3(self.DenseDecoder2(out5))
        return out


class Dense(nn.Module):
    def __init__(self, num_classes, pretrain = True, block_config=((4,4), (12,16,4)), growth_rate =32):
        super(Dense, self).__init__()
        ############# First downsampling  ############## 512

        self.encoder = Encoder(pretrain=pretrain, inter_planes=256, out_planes=256, block_config=block_config[0], growth_rate=growth_rate)
        self.decoder = Decoder(in_planes=256, inter_planes=256, out_planes=32, block_config=block_config[1], growth_rate=growth_rate)
        self.conv_out = nn.Conv2d(in_channels=32, out_channels=num_classes, kernel_size=(3,3), stride=(1,1), padding=1)
    def forward(self, x):
        x0, x1, x2 = self.encoder(x)
        out = self.decoder(x0, x1, x2)
        out = self.conv_out(out)
        return out


class CBAM(nn.Module):

    """Convolutional Block Attention Module

    https://eccv2018.org/openaccess/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf

    """

    def __init__(self, in_channels):

        """
        :param in_channels: int

            Number of input channels.

        """

        super().__init__()

        self.CAM = CAM(in_channels)

        self.SAM = SAM()


    def forward(self, input_tensor):

        # Apply channel attention module

        channel_att_map = self.CAM(input_tensor)

        # Perform elementwise multiplication with channel attention map.

        gated_tensor = torch.mul(input_tensor, channel_att_map)  # (bs, c, h, w) x (bs, c, 1, 1)

        # Apply spatial attention module

        spatial_att_map = self.SAM(gated_tensor)

        # Perform elementwise multiplication with spatial attention map.

        refined_tensor = torch.mul(gated_tensor, spatial_att_map)  # (bs, c, h, w) x (bs, 1, h, w)

        return refined_tensor


class CAM(nn.Module):

    """Channel Attention Module

    """

    def __init__(self, in_channels, reduction_ratio=16):

        """
        :param in_channels: int

            Number of input channels.

        :param reduction_ratio: int

            Channels reduction ratio for MLP.
        """

        super().__init__()

        reduced_channels_num = (in_channels // reduction_ratio) if (in_channels > reduction_ratio) else 1

        pointwise_in = nn.Conv2d(kernel_size=1, in_channels=in_channels, out_channels=reduced_channels_num)

        pointwise_out = nn.Conv2d(kernel_size=1, in_channels=reduced_channels_num, out_channels=in_channels)

        # In the original paper there is a standard MLP with one hidden layer.

        # TODO: try linear layers instead of pointwise convolutions.

        self.MLP = nn.Sequential(pointwise_in, nn.ReLU(), pointwise_out,)

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):

        h, w = input_tensor.size(2), input_tensor.size(3)



        # Get (channels, 1, 1) tensor after MaxPool

        max_feat = F.max_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Get (channels, 1, 1) tensor after AvgPool

        avg_feat = F.avg_pool2d(input_tensor, kernel_size=(h, w), stride=(h, w))

        # Throw maxpooled and avgpooled features into shared MLP

        max_feat_mlp = self.MLP(max_feat)

        avg_feat_mlp = self.MLP(avg_feat)

        # Get channel attention map of elementwise features sum.

        channel_attention_map = self.sigmoid(max_feat_mlp + avg_feat_mlp)

        return channel_attention_map


class SAM(nn.Module):

    """Spatial Attention Module"""



    def __init__(self, ks=7):

        """



        :param ks: int

            kernel size for spatial conv layer.

        """



        super().__init__()

        self.ks = ks

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(kernel_size=self.ks, in_channels=2, out_channels=1)



    def _get_padding(self, dim_in, kernel_size, stride):

        """Calculates \'SAME\' padding for conv layer for specific dimension.



        :param dim_in: int

            Size of dimension (height or width).

        :param kernel_size: int

            kernel size used in conv layer.

        :param stride: int

            stride used in conv layer.

        :return: int

            padding

        """



        padding = (stride * (dim_in - 1) - dim_in + kernel_size) // 2

        return padding



    def forward(self, input_tensor):

        c, h, w = input_tensor.size(1), input_tensor.size(2), input_tensor.size(3)


        # Permute input tensor for being able to apply MaxPool and AvgPool along the channel axis

        permuted = input_tensor.view(-1, c, h * w).permute(0,2,1)

        # Get (1, h, w) tensor after MaxPool

        max_feat = F.max_pool1d(permuted, kernel_size=c, stride=c)

        max_feat = max_feat.permute(0,2,1).view(-1, 1, h, w)


        # Get (1, h, w) tensor after AvgPool

        avg_feat = F.avg_pool1d(permuted, kernel_size=c, stride=c)

        avg_feat = avg_feat.permute(0,2,1).view(-1, 1, h, w)



        # Concatenate feature maps along the channel axis, so shape would be (2, h, w)

        concatenated = torch.cat([max_feat, avg_feat], dim=1)

        # Get pad values for SAME padding for conv2d

        h_pad = self._get_padding(concatenated.shape[2], self.ks, 1)

        w_pad = self._get_padding(concatenated.shape[3], self.ks, 1)

        # Get spatial attention map over concatenated features.

        self.conv.padding = (h_pad, w_pad)

        spatial_attention_map = self.sigmoid(

            self.conv(concatenated)

        )

        return spatial_attention_map

