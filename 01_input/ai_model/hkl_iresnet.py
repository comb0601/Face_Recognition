import torch
from torch import nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, block_size, drop_prob=0.1):
        super(DropBlock2D, self).__init__()

        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        # shape: (bsize, channels, height, width)

        assert x.dim() == 4, \
            "Expected input with 4 dimensions (bsize, channels, height, width)"

        if not self.training or self.drop_prob == 0.:
            return x
        else:
            # get gamma value
            gamma = self._compute_gamma(x)

            # sample mask
            mask = (torch.rand(x.shape[0], *x.shape[2:]) < gamma).float()

            # place mask on input device
            mask = mask.to(x.device)

            # compute block mask
            block_mask = self._compute_block_mask(mask)

            # apply block mask
            out = x * block_mask[:, None, :, :]

            # scale output
            out = out * block_mask.numel() / block_mask.sum()

            return out

    def _compute_block_mask(self, mask):
        block_mask = F.max_pool2d(input=mask[:, None, :, :],
                                  kernel_size=(self.block_size, self.block_size),
                                  stride=(1, 1),
                                  padding=self.block_size // 2)

        if self.block_size % 2 == 0:
            block_mask = block_mask[:, :, :-1, :-1]

        block_mask = 1 - block_mask.squeeze(1)

        return block_mask

    def _compute_gamma(self, x):
        return self.drop_prob / (self.block_size ** 2)

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_c,
                 out_c,
                 stride=1,
                 use_dropblock=False,
                 use_dropblock_in_skip_connection=False
                 ):
        super(IBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act1 = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.dropblock = DropBlock2D(drop_prob=0.1, block_size=7) if use_dropblock else nn.Identity()

        if stride > 1 or in_c != out_c:
            ''' origin shortcut '''
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(out_c),
                DropBlock2D(drop_prob=0.1, block_size=7) if use_dropblock and use_dropblock_in_skip_connection else nn.Identity() 
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.dropblock(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn3(x)

        identity = self.shortcut(identity)

        # print(x.shape)
        # print(identity.shape)

        x += identity

        return x

class RevisedIBasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 in_c,
                 out_c,
                 stride=1,
                 use_dropblock=False,
                 use_dropblock_in_skip_connection=False
                 ):
        super(RevisedIBasicBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_c)
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act1 = nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=stride, padding=1, dilation=1, groups=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.dropblock = DropBlock2D(drop_prob=0.1, block_size=7) if use_dropblock else nn.Identity()

        if stride > 1 or in_c != out_c:
            ''' origin shortcut '''
            # self.shortcut = nn.Sequential(
            #     nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, padding=0, dilation=1, groups=1, bias=False),
            #     nn.BatchNorm2d(out_c)
            # )
            ''' revised shortcut '''
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_c),
                nn.AvgPool2d(2, stride, ceil_mode=True),
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=False),
                nn.BatchNorm2d(out_c),
                DropBlock2D(drop_prob=0.1, block_size=7) if use_dropblock and use_dropblock_in_skip_connection else nn.Identity() 
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        x = self.bn1(x)
        x = self.conv1(x)
        x = self.bn2(x)
        x = self.dropblock(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn3(x)

        identity = self.shortcut(identity)

        # print(x.shape)
        # print(identity.shape)

        x += identity

        return x

class IResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 widths,
                 num_features=512,
                 zero_init_residual=True,
                 input_size=112,
                 use_dropblocks=[0,0,0,0],
                 use_dropout=False,
                 use_dropblock_in_skip_connection=False,
                 fp16=False
                 ):
        super(IResNet, self).__init__()
        
        self.fp16 = fp16

        fc_size = input_size
        for _ in range(4):
            if fc_size % 2 == 0:
                fc_size = fc_size // 2
            else:
                fc_size = (fc_size + 1) // 2
        fc_size = fc_size ** 2

        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.PReLU(widths[0])
        )

        self.body = nn.Sequential(
            self._make_layer(
                block, widths[0], widths[1], layers[0], stride=2, 
                use_dropblock=use_dropblocks[0], use_dropblock_in_skip_connection=use_dropblock_in_skip_connection
            ),
            self._make_layer(
                block, widths[1], widths[2], layers[1], stride=2, 
                use_dropblock=use_dropblocks[1], use_dropblock_in_skip_connection=use_dropblock_in_skip_connection
            ),
            self._make_layer(
                block, widths[2], widths[3], layers[2], stride=2, 
                use_dropblock=use_dropblocks[2], use_dropblock_in_skip_connection=use_dropblock_in_skip_connection
            ),
            self._make_layer(
                block, widths[3], widths[4], layers[3], stride=2, 
                use_dropblock=use_dropblocks[3], use_dropblock_in_skip_connection=use_dropblock_in_skip_connection
            )
        )

        dropout = nn.Dropout(p=0.4, inplace=True) if use_dropout else nn.Identity()

        # self.pre_head = nn.Sequential(
        #     nn.BatchNorm2d(widths[4]),
        #     Flatten(),
        #     dropout
        # )
        # self.head = nn.Sequential(
        #     nn.Linear(widths[4] * fc_size, num_features),
        #     nn.BatchNorm1d(num_features)
        # )

        self.head = nn.Sequential(
            nn.BatchNorm2d(widths[4]),
            Flatten(),
            dropout,
            nn.Linear(widths[4] * fc_size, num_features),
            nn.BatchNorm1d(num_features)
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight,
                #                         mode='fan_out',
                #                         nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock) or isinstance(m, RevisedIBasicBlock):
                    nn.init.constant_(m.bn3.weight, 0)


    def _make_layer(self, block, in_c, out_c, blocks, stride=1, use_dropblock=False, use_dropblock_in_skip_connection=False):
        layers = []

        layers.append(
            block(in_c, out_c, stride, use_dropblock, use_dropblock_in_skip_connection)
        )
        for _ in range(1, blocks):
            layers.append(
                block(out_c, out_c, 1, use_dropblock, use_dropblock_in_skip_connection)
            )

        return nn.Sequential(*layers)

    def set_dropblock_prob(self, drop_prob):
        for m in self.modules():
            if isinstance(m, DropBlock2D):
                m.drop_prob = drop_prob
    
    def set_dropout_prob(self, drop_prob):
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop_prob
    
    def forward(self, x, do_mid_val=False):
        with torch.cuda.amp.autocast(self.fp16):
            x = self.stem(x)
            x = self.body(x)
            # x = self.pre_head(x)
            x = self.head[0](x)
            x = self.head[1](x)
            x = self.head[2](x)
        
        if do_mid_val:
            mid_val = x

        x = self.head[3](x.float() if self.fp16 else x)
        x = self.head[4](x)

        if do_mid_val:
            return mid_val, x
        else:
            return x


def iresnet34(input_size=112):
    return IResNet(IBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size)
def iresnet34_dropout(input_size=112):
    return IResNet(IBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size, use_dropout=True)
def iresnet34_dropblock(input_size=112):
    return IResNet(IBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1])
def iresnet34_dropblock_also_skip_connection(input_size=112):
    return IResNet(IBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropblock_in_skip_connection=True)
def iresnet34_dropblock_dropout(input_size=112):
    return IResNet(IBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=True)

def iresnet34_revised_for_bagnet(input_size=32):
    return IResNet(RevisedIBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size)

def iresnet34_revised(input_size=112):
    return IResNet(RevisedIBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size)
def iresnet34_revised_dropblock_dropout(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3,4,6,3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=True
    )

def iresnet50_revised(input_size=112, **kwargs):
    return IResNet(
        RevisedIBasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], 
        input_size=input_size, **kwargs
    )
def iresnet50_revised_dropout(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[0,0,0,0], use_dropout=True
    )

def iresnet50_revised_emb128(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], 
        num_features=128, input_size=input_size, use_dropblocks=[0,0,0,0], use_dropout=False
    )

def iresnet50_revised_dropblock_dropout(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=True
    )

def iresnet50_revised_dropblock(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 4, 14, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=False
    )

def iresnet100_revised(input_size=112):
    return IResNet(RevisedIBasicBlock, [3, 13, 30, 3], [64, 64, 128, 256, 512], num_features=512, input_size=input_size)

def iresnet200_revised_dropblock(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 30, 63, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=False
    )

def iresnet200_revised_dropblock_dropout(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 30, 63, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size, use_dropblocks=[1,1,1,1], use_dropout=True
    )
def iresnet200_revised(input_size=112, **kwargs):
    return IResNet(
        RevisedIBasicBlock, [3, 30, 63, 3], [64, 64, 128, 256, 512], 
        input_size=input_size, **kwargs
    )

def iresnet200_revised_36GMac(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 30, 63, 3], [78, 78, 156, 312, 624],
        num_features=512, input_size=input_size
    )

def iresnet300_revised(input_size=112):
    return IResNet(
        RevisedIBasicBlock, [3, 45, 98, 3], [64, 64, 128, 256, 512], 
        num_features=512, input_size=input_size
    )

def iresnet154_revised(input_size=112, **kwargs):
    return IResNet(
        RevisedIBasicBlock, [3, 20, 50, 3], [64, 64, 128, 256, 512], 
        input_size=input_size, **kwargs
    )

def iresnet304_revised(input_size=112, **kwargs):
    return IResNet(
        RevisedIBasicBlock, [3, 70, 75, 3], [64, 64, 128, 256, 512], 
        input_size=input_size, **kwargs
    )


def scale(alpha=0, s=2, d_w_r = [0.5, 1, 0.5]):
    e_d = (1-alpha) * d_w_r[0]
    e_w = alpha * d_w_r[1]
    e_r = (1-alpha) * d_w_r[2]

    s_d = s ** (e_d)
    s_w = s ** (e_w / 2)
    s_r = s ** (e_r / 2)

    print(f's_d: {round(s_d,2)}, s_w: {round(s_w,2)}, s_r: {round(s_r,2)}')

    return s_d, s_w, s_r

def scaled_iresnet(depths, widths, input_size=112, **kwargs):
    return IResNet(RevisedIBasicBlock, depths, widths, input_size=input_size, **kwargs)

if __name__ == '__main__':

    default_depths = [3, 30, 63, 3]
    default_widths = [64,64,128,256,512]
    default_input_size = 112
    group_width = 1

    s_d, s_w, s_r = scale(alpha=1.0, s=1.5, d_w_r=[0, 1, 1])

    scaled_depths = [int(d * s_d) for d in default_depths]
    scaled_widths = [int(w * s_w) for w in default_widths]
    scaled_input_size = int(default_input_size * s_r)
    scaled_group_width = int(group_width * s_w)

    scaled_iresnet = scaled_iresnet(scaled_depths, scaled_widths, scaled_input_size)

    from ptflops import get_model_complexity_info

    scaled_iresnet.eval()
    
    with torch.no_grad():
        with torch.cuda.device(0):
            macs, params = get_model_complexity_info(scaled_iresnet, (3, scaled_input_size, scaled_input_size), as_strings=True, print_per_layer_stat=False, verbose=True)
            print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
            print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    print(f'scaled: d - {scaled_depths}, w - {scaled_widths}, r - {scaled_input_size}, g - {scaled_group_width}')

    
    # x = torch.ones([64,3,112,112])
    # net = iresnet34(112)
    # y = net(x)

    # print(y.shape)
