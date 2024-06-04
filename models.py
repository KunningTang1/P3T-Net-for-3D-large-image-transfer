import torch.nn as nn
import torch.nn.functional as F
import torch
import functools

 #%%
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs=5, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, outermost=False,use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, outermost=False,norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, outermost=False,norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block,  outermost=False,norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            # up = [uprelu, upconv, upnorm]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc*2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
            # return x+self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)
            # return x+self.model(x)

 #%% 
class ResnetGenerator1(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=5, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator1, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.Upsample(scale_factor=2,mode='nearest'),
                      nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=1,
                                         padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
    
    
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=5, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            # model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
            #                               kernel_size=3, stride=2,
            #                               padding=1, output_padding=1,
            #                               bias=use_bias),
            #           norm_layer(int(ngf * mult / 2)),
            #           nn.ReLU(True)]
            model += [nn.Upsample(scale_factor=2,mode='nearest'),
              nn.Conv2d(ngf * mult, int(ngf * mult / 2),
                                  kernel_size=3, stride=1,
                                  padding=1,
                                  bias=use_bias),
              norm_layer(int(ngf * mult / 2)),
              nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetGenerator_transpose(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=4, padding_type='reflect'):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator_transpose, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block
        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.
        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not
        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

     #%%
class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=4, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)
    
# Defines the PatchGAN discriminator with the specified arguments.
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.main = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # nn.Conv2d(256, 512, 4, padding=1),
            # nn.InstanceNorm2d(512),
            # nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, 4, padding=1),
        )

    def forward(self, x):
        x = self.main(x)
        x = F.avg_pool2d(x, x.size()[2:])
        # x = torch.flatten(x, 1)
        x = torch.flatten(x)
        return x

import torch.nn as nn


def unet_conv(in_planes, out_planes):
    conv = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
        nn.Conv2d(out_planes, out_planes, 3, 1, 1),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(False),
        # nn.Dropout(0.2)
    )
    return conv


class Uresnet_G(nn.Module):
    def __init__(self, input_nbr = 1):
        super(Uresnet_G, self).__init__()
        
        # forwarf
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )      # No.1 long skip 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,1,1),
            nn.ReLU(True),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3,1,1),
            nn.ReLU(True),
        )      # No3 resudual block
        
        self.downconv7 = unet_conv(512, 512) # No4 long skip
        
        
        self.updeconv2 = nn.Sequential(
            # nn.ConvTranspose2d(512, 256, 2, 2),
            nn.ConvTranspose2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
           
        self.upconv3 = nn.Sequential(
            nn.Conv2d(512, 256, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(False),   
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, 1, 1)  # 6 is number of phases to be segmented
        
        # self.fc_params = nn.Sequential (
        #     nn.Linear(512*4*4, 512),
        #     nn.BatchNorm1d(512),
        #     )

        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(512, 10)
        #     )

    def forward(self, x):
        
        # encoding
        x1 = self.downconv1(x) 
 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 += x3
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 += x6
        x8 = self.maxpool(x7)
        
        x9 = self.downconv6(x8)
        x10 = self.downconv7(x9)
        x10 += x9
        y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2,align_corners=True)
        y4 = self.updeconv2(y3)
        y5 = self.upconv3(torch.cat([y4, x7],1))
        y6 = self.upconv4(y5)
        y6 += y5
        
        y6 = nn.functional.interpolate(y6, mode='bilinear', scale_factor=2,align_corners=True)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 += y8
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2,align_corners=True)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 += y11
     
        out = self.last(y12)
        
        return out

def uresnet_G():
    net = uresnet_G()
    return net    

class Uresnet_G1(nn.Module):
    def __init__(self, input_nbr = 1):
        super(Uresnet_G1, self).__init__()
        
        # forwarf
        self.downconv1 = nn.Sequential(
            nn.Conv2d(input_nbr, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )      # No.1 long skip 
        
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
        )      # No1 resudual block
        
        self.downconv3 = unet_conv(128, 128) # No2 long skip
        self.maxpool = nn.MaxPool2d(2, 2)
        
        self.downconv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3,1,1),
            nn.ReLU(True),
        )      # No2 resudual block
        
        self.downconv5 = unet_conv(256, 256) # No3 long skip
        # self.maxpool = nn.MaxPool2d(2, 2)
        
        # self.downconv6 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3,1,1),
        #     nn.ReLU(True),
        # )      # No3 resudual block
        
        # self.downconv7 = unet_conv(512, 512) # No4 long skip
        
        
        # self.updeconv2 = nn.Sequential(
        #     # nn.ConvTranspose2d(512, 256, 2, 2),
        #     nn.ConvTranspose2d(512, 256, 3, 1, 1),
        #     nn.BatchNorm2d(256),
        # )
           
        # self.upconv3 = nn.Sequential(
        #     nn.Conv2d(512, 256, 1),
        #     nn.ReLU(False),
        # )       # No6 resudual block
        # self.upconv4 = unet_conv(256, 256)
        
        self.updeconv3 = nn.Sequential(
            # nn.ConvTranspose2d(256, 128, 2, 2),
            nn.ConvTranspose2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
           
        self.upconv5 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.ReLU(False),
        )       # No6 resudual block
        self.upconv6 = unet_conv(128, 128)
        self.updeconv4 = nn.Sequential(
            # nn.ConvTranspose2d(128, 64, 2, 2),
            nn.ConvTranspose2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )
        
        self.upconv7 = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.ReLU(False),   
        )       # No6 resudual block
        self.upconv8 = unet_conv(64, 64)
        
        self.last = nn.Conv2d(64, 1, 1)  # 6 is number of phases to be segmented
        
        # self.fc_params = nn.Sequential (
        #     nn.Linear(512*4*4, 512),
        #     nn.BatchNorm1d(512),
        #     )

        # self.classifier = nn.Sequential(
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(512, 10)
        #     )

    def forward(self, x):
        
        # encoding
        x1 = self.downconv1(x) 
 
        x2 = self.maxpool(x1)     
        x3 = self.downconv2(x2)
        x4 = self.downconv3(x3)      
        x4 += x3
        x5 = self.maxpool(x4)
        
        x6 = self.downconv4(x5)
        x7 = self.downconv5(x6)
        x7 += x6
        # x8 = self.maxpool(x7)
        
        # x9 = self.downconv6(x8)
        # x10 = self.downconv7(x9)
        # x10 += x9
        # y3 = nn.functional.interpolate(x10, mode='bilinear', scale_factor=2,align_corners=True)
        # y4 = self.updeconv2(y3)
        # y5 = self.upconv3(torch.cat([y4, x7],1))
        # y6 = self.upconv4(y5)
        # y6 += y5
        
        y6 = nn.functional.interpolate(x7, mode='bilinear', scale_factor=2,align_corners=True)
        y7 = self.updeconv3(y6)   
        y8 = self.upconv5(torch.cat([y7, x4],1))
        y9 = self.upconv6(y8)
        y9 += y8
        
        y9 = nn.functional.interpolate(y9, mode='bilinear', scale_factor=2,align_corners=True)
        y10= self.updeconv4(y9)
        y11 = self.upconv7(torch.cat([y10, x1],1))
        y12 = self.upconv8(y11)
        y12 += y11
     
        out = self.last(y12)
        
        return out

def uresnet_G1():
    net = uresnet_G1()
    return net    
class FeatureDiscriminator(nn.Module):
    def __init__(self):
        super(FeatureDiscriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(10, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2),
        )

    def forward(self, score):
        out = self.discriminator(score)
        return out

















