from model_mamba import videomamba_tiny
from model_transformer import *
from model_detectron import *

class Mamba_cross(nn.Module):
    def __init__(self, args):
        super(Mamba_cross, self).__init__()

        top_block = LastLevelP6(256, 256, "p5")

        self.mamba = videomamba_tiny(num_frames=15, num_classes=args.num_c)
        self.swintrans = Transformer_Det(args)

        self.stem_fea = nn.Sequential(DoubleConv(args.embed_dim * 4, 128, 2),
                                      DoubleConv(128, 128, 1),
                                      DoubleConv(128, 120, 2)
                                      )

        self.stem_cc = nn.Sequential(DoubleConv(3, 8, 2),
                                      DoubleConv(8, 8, 1),
                                      DoubleConv(8, 8, 2)
                                      )


        self.fpn_with_backbone = FPN(VoVNet_Det(["stage3", "stage4", "stage5"]),
                                     ["stage3", "stage4", "stage5"], 256, top_block=top_block)

        input_shape = {
            "p3": ShapeSpec(channels=256, height=32, width=32),
            "p4": ShapeSpec(channels=256, height=16, width=16),
            "p5": ShapeSpec(channels=256, height=8, width=8),
            "p6": ShapeSpec(channels=256, height=4, width=4)
        }

        self.fcos = FCOS(args, input_shape)

        self.if_concat_cc = args.if_concat_cc
        self.if_mamba = args.if_mamba


        if args.fix_trans:
            for param in self.swintrans.parameters():
                param.requires_grad = False
            for param in self.stem_fea.parameters():
                param.requires_grad = False
            for param in self.stem_cc.parameters():
                param.requires_grad = False
            for param in self.fpn_with_backbone.parameters():
                param.requires_grad = False

        if args.fix_mamba:
            for param in self.mamba.parameters():
                param.requires_grad = False



    def forward(self, x):

        x_ctp = x['image_ctp']

        if self.if_mamba:
            x_m = self.mamba(x_ctp)
            if torch.isnan(x_m).any() or torch.isinf(x_m).any():
                print("NaN or Inf detected in xm")
        else:
            x_m = x['image_0']

        x_para_list = [x['image_41'], x['image_42'], x['image_43'], x['image_44']]

        x_cc = x['image_3']

        if self.if_concat_cc:
            x_i = self.swintrans(x_m, x_para_list, x_cc)
        else:
            x_i = self.swintrans(x_m, x_para_list)


        x_i = x_i.contiguous()
        x_i = self.stem_fea(x_i)

        x_cc = self.stem_cc(x_cc)

        x_i = torch.cat([x_i, x_cc], dim=1)

        fpn_features = self.fpn_with_backbone(x_i)
        proposals, loss = self.fcos(x_i, fpn_features, x)

        return proposals, loss

        return x_out

