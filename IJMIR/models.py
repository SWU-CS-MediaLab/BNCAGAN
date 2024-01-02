# -*- codign:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import functools
import common
from tools import extract_image_patches, \
    reduce_mean, reduce_sum, same_padding
from torch.nn.parameter import Parameter

class Generator(nn.Module):
    """Generator network"""

    def __init__(self, conv_dim, norm_fun, act_fun, use_sn):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=False)

        self.enc1 = ConvBlock(in_channels=3, out_channels=conv_dim * 1, kernel_size=7, stride=1, padding=0, dilation=1,
                              use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)

        self.enc2 = ConvBlock(in_channels=conv_dim * 1, out_channels=conv_dim * 2, kernel_size=3, stride=2, padding=0,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)

        self.enc3 = ConvBlock(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=3, stride=2, padding=0,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)

        self.enc4 = ConvBlock(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=3, stride=2, padding=0,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)

        self.enc5 = ConvBlock(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=3, stride=2, padding=0,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)


        channels_mid = int(conv_dim * 4)
        self.channels_cond = conv_dim * 8

        self.relu = nn.LeakyReLU(inplace=True)

        self.upsample0 = nn.Sequential(Interpolate(2, 'bilinear', True),
                                       SNConv(conv_dim * 32, conv_dim * 16, 1, 1, 0, 1, True, use_sn))
        self.upsample1 = nn.Sequential(Interpolate(2, 'bilinear', True),
                                       SNConv(conv_dim * 16, conv_dim * 8, 1, 1, 0, 1, True, use_sn))
        self.upsample2 = nn.Sequential(Interpolate(2, 'bilinear', True),
                                       SNConv(conv_dim * 8, conv_dim * 4, 1, 1, 0, 1, True, use_sn))
        self.upsample3 = nn.Sequential(Interpolate(2, 'bilinear', True),
                                       SNConv(conv_dim * 4, conv_dim * 2, 1, 1, 0, 1, True, use_sn))
        self.upsample4 = nn.Sequential(Interpolate(2, 'bilinear', True),
                                       SNConv(conv_dim * 2, conv_dim * 1, 1, 1, 0, 1, True, use_sn))

        self.dec0 = ConvBlock(in_channels=conv_dim * 32, out_channels=conv_dim * 16, kernel_size=3, stride=1, padding=0,
                              dilation=1, use_bias=True, norm_fun='ILN', act_fun=act_fun, use_sn=use_sn)

        self.dec1 = ConvBlock(in_channels=conv_dim * 16, out_channels=conv_dim * 8, kernel_size=3, stride=1, padding=0,
                              dilation=1, use_bias=True, norm_fun='ILN', act_fun=act_fun, use_sn=use_sn)

        self.dec2 = ConvBlock(in_channels=conv_dim * 8, out_channels=conv_dim * 4, kernel_size=3, stride=1, padding=0,
                              dilation=1, use_bias=True, norm_fun='ILN', act_fun=act_fun, use_sn=use_sn)

        self.dec3 = ConvBlock(in_channels=conv_dim * 4, out_channels=conv_dim * 2, kernel_size=3, stride=1, padding=0,
                              dilation=1, use_bias=True, norm_fun='ILN', act_fun=act_fun, use_sn=use_sn)

        self.dec4 = ConvBlock(in_channels=conv_dim * 2, out_channels=conv_dim * 1, kernel_size=3, stride=1, padding=0,
                              dilation=1, use_bias=True, norm_fun='ILN', act_fun=act_fun, use_sn=use_sn)

        self.dec5 = nn.Sequential(
            SNConv(in_channels=conv_dim * 1, out_channels=conv_dim * 1, kernel_size=3, stride=1, padding=0, dilation=1,
                   use_bias=True, use_sn=False),
            SNConv(in_channels=conv_dim * 1, out_channels=3, kernel_size=7, stride=1, padding=0, dilation=1,
                   use_bias=True, use_sn=False),
            nn.Tanh()
        )


        self.logit_fc = nn.Linear(conv_dim * 8, 1, bias=False)

        FC1 = [nn.Linear(conv_dim * 8, conv_dim * 8, bias=False),
              nn.ReLU(True),
              nn.Linear(conv_dim * 8, conv_dim * 8, bias=False),
              nn.ReLU(True)]
        self.FC1 = nn.Sequential(*FC1)

        FC2 = [nn.Linear(conv_dim * 8, conv_dim * 8, bias=False),
              nn.ReLU(True),
              nn.Linear(conv_dim * 8, conv_dim * 8, bias=False),
              nn.ReLU(True)]
        self.FC2 = nn.Sequential(*FC2)

        self.gamma1 = nn.Linear(conv_dim * 8, conv_dim * 8, bias=False)
        self.beta1 = nn.Linear(conv_dim * 8, conv_dim * 8, bias=False)
        self.gap_fc1 = nn.Linear(conv_dim * 8, 1, bias=False)
        self.gmp_fc1 = nn.Linear(conv_dim * 8, 1, bias=False)
        self.gamma2 = nn.Linear(conv_dim * 8, conv_dim * 8, bias=False)
        self.beta2 = nn.Linear(conv_dim * 8, conv_dim * 8, bias=False)
        self.gap_fc2 = nn.Linear(conv_dim * 8, 1, bias=False)
        self.gmp_fc2 = nn.Linear(conv_dim * 8, 1, bias=False)


        self.ResnetBlock = AdaILNBlock(conv_dim * 8, use_bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        reduction_ratio = 16
        self.fc1 = nn.Conv2d(conv_dim * 8, conv_dim * 8 // reduction_ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(conv_dim * 8 // reduction_ratio, conv_dim * 8, 1, bias=False)
        self.fc3 = nn.Conv2d(conv_dim * 8 // reduction_ratio, conv_dim * 8, 1, bias=False)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)
        self.max_pool2 = nn.AdaptiveMaxPool2d(1)
        self.fc_end = nn.Linear(conv_dim * 8, 1, bias=False) #

        self.sigmoid = nn.Sigmoid()

        self.conv1x1 = nn.Conv2d(in_channels=conv_dim * 48, out_channels=conv_dim * 8, kernel_size=1, bias=False)
        self.reconv1x1 = nn.Conv2d(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=1)

        self.lin1 = nn.AdaptiveAvgPool2d(1)
        self.lin2 = nn.Linear(conv_dim * 8, 64, bias=False)
        self.lin12 = nn.AdaptiveAvgPool2d(1)
        self.lin22 = nn.Linear(conv_dim * 8, 64, bias=False)

    def forward(self, raw, exp, mode):

        if mode == 'train':

            raw1 = self.enc1(raw) 
            raw2 = self.enc2(raw1) 
            raw3 = self.enc3(raw2) 
            raw4 = self.enc4(raw3) 

            raw_lin = self.lin1(raw4)

            raw_lin = self.lin2(raw_lin.view(raw4.shape[0], -1))

            exp1 = self.enc1(exp)  
            exp2 = self.enc2(exp1) 
            exp3 = self.enc3(exp2)  
            exp4 = self.enc4(exp3)  

            exp_lin = self.lin1(exp4)
            exp_lin = self.lin2(exp_lin.view(exp4.shape[0], -1))

            raw_avg_out = self.avg_pool(raw4) 
            raw_max_out = self.max_pool(raw4)

            
            raw_out_z = raw_avg_out + raw_max_out

            raw_out_z = self.softmax(raw_out_z)

            raw_out = raw4 * raw_out_z  

            raw_gap_logit = self.gap_fc1(raw_avg_out.view(raw4.shape[0], -1))  
            raw_gmp_logit = self.gmp_fc1(raw_max_out.view(raw4.shape[0], -1))

            raw_logit = torch.cat([raw_gap_logit, raw_gmp_logit], 1)

            raw_x_ = torch.nn.functional.adaptive_avg_pool2d(raw_out, 1)
            raw_x_ = self.FC1(raw_x_.view(raw_x_.shape[0], -1))  
            raw_gamma, raw_beta = self.gamma1(raw_x_), self.beta1(raw_x_)  

            exp_avg_out = self.avg_pool(exp4) 
            exp_max_out = self.max_pool(exp4)
            exp_out_z = exp_avg_out + exp_max_out

            exp_out_z = self.softmax(exp_out_z)
            exp_out = exp4 * exp_out_z  # [2, 256, 32, 32]

            exp_gap_logit = self.gap_fc2(exp_avg_out.view(exp4.shape[0], -1)) 
            exp_gmp_logit = self.gmp_fc2(exp_max_out.view(exp4.shape[0], -1))

            exp_logit = torch.cat([exp_gap_logit, exp_gmp_logit], 1)

            exp_x_ = torch.nn.functional.adaptive_avg_pool2d(exp_out, 1)      
            exp_x_ = self.FC2(exp_x_.view(exp_x_.shape[0], -1))
            exp_gamma, exp_beta = self.gamma2(exp_x_), self.beta2(exp_x_) 

            raw_res1, res1 = self.ResnetBlock(raw_out, exp_gamma, exp_beta)
            raw_res2, res2 = self.ResnetBlock(raw_res1, exp_gamma, exp_beta)
            raw_res3, res3 = self.ResnetBlock(raw_res2, exp_gamma, exp_beta)
            raw_res4, res4 = self.ResnetBlock(raw_res3, exp_gamma, exp_beta)
            raw_res5, res5 = self.ResnetBlock(raw_res4, exp_gamma, exp_beta)
            _, res6 = self.ResnetBlock(raw_res5, exp_gamma, exp_beta)

            res_all = torch.cat([res1, res2, res3, res4, res5, res6], dim=1)

            res_all = self.conv1x1(res_all) + raw_out
            exp_res1, res12 = self.ResnetBlock(exp_out, raw_gamma, raw_beta)
            exp_res2, res22 = self.ResnetBlock(exp_res1, raw_gamma, raw_beta)
            exp_res3, res32 = self.ResnetBlock(exp_res2, raw_gamma, raw_beta)
            exp_res4, res42 = self.ResnetBlock(exp_res3, raw_gamma, raw_beta)
            exp_res5, res52 = self.ResnetBlock(exp_res4, raw_gamma, raw_beta)
            _, res62 = self.ResnetBlock(exp_res5, raw_gamma, raw_beta)

            res_all2 = torch.cat([res12, res22, res32, res42, res52, res62], dim=1)
            res_all2 = self.conv1x1(res_all2) + exp_out

            raw_y2 = self.upsample2(res_all)  # [2, 128, 64, 64]

            raw_y2 = torch.cat([raw_y2, raw3], dim=1)  # [2, 256, 64, 64]
            raw_y2 = self.dec2(raw_y2)  # [2, 128, 64, 64]

            raw_y3 = self.upsample3(raw_y2)  # [2, 64, 128, 128]
            raw_y3 = torch.cat([raw_y3, raw2], dim=1)  # [2, 128, 128, 128]
            raw_y3 = self.dec3(raw_y3)  # [2, 64, 128, 128]

            raw_y4 = self.upsample4(raw_y3)  # [2, 32, 256, 256]
            raw_x1_avg = torch.mean(raw1, dim=1, keepdim=True)
            raw_x1_max, _ = torch.max(raw1, dim=1, keepdim=True)
            raw_x1_out = raw_x1_avg + raw_x1_max

            raw_fout = raw_y4 * raw_x1_out  # [2, 32, 256, 256]
            raw_fout = self.dec5(raw_fout)  # [2, 3, 256, 256]

            raw_fout = torch.clamp((raw_fout + raw), min=-1.0, max=1.0)


            exp_y2 = self.upsample2(res_all2)  # [2, 128, 64, 64]
            exp_y2 = torch.cat([exp_y2, exp3], dim=1)  # [2, 256, 64, 64]
            exp_y2 = self.dec2(exp_y2)  # [2, 128, 64, 64]

            exp_y3 = self.upsample3(exp_y2)  # [2, 64, 128, 128]
            exp_y3 = torch.cat([exp_y3, exp2], dim=1)  # [2, 128, 128, 128]
            exp_y3 = self.dec3(exp_y3)  # [2, 64, 128, 128]

            exp_y4 = self.upsample4(exp_y3)  # [2, 32, 256, 256]
            exp_x1_avg = torch.mean(exp1, dim=1, keepdim=True)
            exp_x1_max, _ = torch.max(exp1, dim=1, keepdim=True)
            exp_x1_out = exp_x1_avg + exp_x1_max
            exp_fout = exp_y4 * exp_x1_out  # [2, 32, 256, 256]
            exp_fout = self.dec5(exp_fout)  # [2, 3, 256, 256]

            exp_fout = torch.clamp((exp_fout + exp), min=-1.0, max=1.0)

            return raw_fout, raw_logit, raw_lin, exp_fout, exp_logit, exp_lin
        if mode == 'test':

            raw1 = self.enc1(raw)  # x1 : [1, 32, 256, 256]
            raw2 = self.enc2(raw1)  # x2 : [1, 64, 128, 128]
            raw3 = self.enc3(raw2)  # x3 : [1, 128, 64, 64]
            raw4 = self.enc4(raw3)  # x4 : [1, 256, 32, 32]
            raw_lin = self.lin1(raw4)
            raw_lin = self.lin2(raw_lin.view(raw4.shape[0], -1))


            raw_avg_out = self.avg_pool(raw4) 
            raw_max_out = self.max_pool(raw4)
            raw_out_z = raw_avg_out + raw_max_out

            raw_out_z = self.softmax(raw_out_z)
            raw_out = raw4 * raw_out_z 

            raw_gap_logit = self.gap_fc1(raw_avg_out.view(raw4.shape[0], -1)) 
            raw_gmp_logit = self.gmp_fc1(raw_max_out.view(raw4.shape[0], -1))
            raw_logit = torch.cat([raw_gap_logit, raw_gmp_logit], 1)

            raw_x_ = torch.nn.functional.adaptive_avg_pool2d(raw_out, 1)  
            raw_x_ = self.FC1(raw_x_.view(raw_x_.shape[0], -1)) 
            raw_gamma, raw_beta = self.gamma1(raw_x_), self.beta1(raw_x_) 

            raw_res1, res1 = self.ResnetBlock(raw_out, raw_gamma, raw_beta)
            raw_res2, res2 = self.ResnetBlock(raw_res1, raw_gamma, raw_beta)
            raw_res3, res3 = self.ResnetBlock(raw_res2, raw_gamma, raw_beta)
            raw_res4, res4 = self.ResnetBlock(raw_res3, raw_gamma, raw_beta)
            raw_res5, res5 = self.ResnetBlock(raw_res4, raw_gamma, raw_beta)
            _, res6 = self.ResnetBlock(raw_res5, raw_gamma, raw_beta)

            res_all = torch.cat([res1, res2, res3, res4, res5, res6], dim=1)
            res_all = self.conv1x1(res_all) + raw_out

            raw_y2 = self.upsample2(res_all)  # [2, 128, 64, 64]
            raw_y2 = torch.cat([raw_y2, raw3], dim=1)  # [2, 256, 64, 64]
            raw_y2 = self.dec2(raw_y2)  # [2, 128, 64, 64]

            raw_y3 = self.upsample3(raw_y2)  # [2, 64, 128, 128]
            raw_y3 = torch.cat([raw_y3, raw2], dim=1)  # [2, 128, 128, 128]
            raw_y3 = self.dec3(raw_y3)  # [2, 64, 128, 128]

            raw_y4 = self.upsample4(raw_y3)  # [2, 32, 256, 256]
            raw_x1_avg = torch.mean(raw1, dim=1, keepdim=True)
            raw_x1_max, _ = torch.max(raw1, dim=1, keepdim=True)
            raw_x1_out = raw_x1_avg + raw_x1_max

            raw_fout = raw_y4 * raw_x1_out  # [2, 32, 256, 256]
            raw_fout = self.dec5(raw_fout)  # [2, 3, 256, 256]

            raw_fout = torch.clamp((raw_fout + raw), min=-1.0, max=1.0)

            return raw_fout, raw_logit, raw_lin, None, None, None


class SNConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, use_sn):
        super(SNConv, self).__init__()
        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.main = nn.Sequential(

            nn.ReflectionPad2d(self.padding),
            SpectralNorm(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=0, dilation=dilation, bias=use_bias), use_sn),
        )

    def forward(self, x):
        return self.main(x)



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun,
                 use_sn):
        super(ConvBlock, self).__init__()

        self.padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        main = []
        main.append(nn.ReflectionPad2d(self.padding))
        main.append(SpectralNorm(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0, dilation=dilation, bias=use_bias), use_sn))
        norm_fun = get_norm_fun(norm_fun)
        main.append(norm_fun(out_channels))
        main.append(get_act_fun(act_fun))
        self.main = nn.Sequential(*main)

    def forward(self, x):
        return self.main(x)


class Discriminator(nn.Module):
    def __init__(self, conv_dim, norm_fun, act_fun, use_sn, adv_loss_type):
        super(Discriminator, self).__init__()

        d_1 = [dis_conv_block(in_channels=3, out_channels=conv_dim, kernel_size=7, stride=2, padding=3, dilation=1,
                              use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_1_pred = [
            dis_pred_conv_block(in_channels=conv_dim, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1,
                                use_bias=False, type=adv_loss_type)]

        d_2 = [dis_conv_block(in_channels=conv_dim, out_channels=conv_dim * 2, kernel_size=7, stride=2, padding=3,
                              dilation=1, norm_fun=norm_fun, use_bias=True, act_fun=act_fun, use_sn=use_sn)]

        d_3 = [dis_conv_block(in_channels=conv_dim * 2, out_channels=conv_dim * 4, kernel_size=7, stride=2, padding=3,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_3_pred = [dis_pred_conv_block(in_channels=conv_dim * 4, out_channels=1, kernel_size=7, stride=1, padding=3,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        d_3_three_pred = [dis_pred_conv_block(in_channels=3, out_channels=1, kernel_size=7, stride=1, padding=3,
                                              dilation=1, use_bias=False, type=adv_loss_type)]
        self.conv1x1 = nn.Conv2d(in_channels=conv_dim * 4, out_channels=3, kernel_size=1, bias=False)
        self.fconv1x1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
        self.gconv1x1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)
        self.hconv1x1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=1, bias=False)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

        d_4 = [dis_conv_block(in_channels=conv_dim * 4, out_channels=conv_dim * 8, kernel_size=5, stride=2, padding=2,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]

        d_5 = [dis_conv_block(in_channels=conv_dim * 8, out_channels=conv_dim * 16, kernel_size=5, stride=2, padding=2,
                              dilation=1, use_bias=True, norm_fun=norm_fun, act_fun=act_fun, use_sn=use_sn)]
        d_5_pred = [dis_pred_conv_block(in_channels=conv_dim * 16, out_channels=1, kernel_size=5, stride=1, padding=2,
                                        dilation=1, use_bias=False, type=adv_loss_type)]

        self.d1 = nn.Sequential(*d_1)
        self.d1_pred = nn.Sequential(*d_1_pred)
        self.d2 = nn.Sequential(*d_2)

        self.d3 = nn.Sequential(*d_3)
        self.d3_pred = nn.Sequential(*d_3_pred)
        self.d3_three_pred = nn.Sequential(*d_3_three_pred)
        self.d4 = nn.Sequential(*d_4)

        self.d5 = nn.Sequential(*d_5)
        self.d5_pred = nn.Sequential(*d_5_pred)

    def forward(self, x):

        ds1 = self.d1(x)  
        ds1_pred = self.d1_pred(ds1)  

        ds2 = self.d2(ds1)  
        # ds2_pred = self.d2_pred(ds2)  

        ds3 = self.d3(ds2)  
        ds3_pred = self.d3_pred(ds3)  

        ds3_three = self.conv1x1(ds3)  
        m_batchsize, C, width, height = ds3_three.size()
        ds3_f = self.fconv1x1(ds3_three).view(m_batchsize, -1, width * height).permute(0, 2, 1)  
        ds3_g = self.gconv1x1(ds3_three).view(m_batchsize, -1, width * height) 

        attention = self.softmax(torch.bmm(ds3_g, ds3_f))  
        
        ds3_h = self.hconv1x1(ds3_three).view(m_batchsize, -1, width * height)  
        out = torch.bmm(attention.permute(0, 2, 1), ds3_h)  
        out = out.view(m_batchsize, C, width, height)  

        ds3_out = self.gamma * out + ds3_three
        ds3_three_pred = self.d3_three_pred(ds3_out)  

        ds4 = self.d4(ds3)  

        ds5 = self.d5(ds4)  
        ds5_pred = self.d5_pred(ds5)  

        return [ds1_pred, ds3_pred, ds5_pred, ds3_three_pred]

def dis_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, norm_fun, act_fun,
                   use_sn):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []

    main.append(nn.ReflectionPad2d(padding))
    main.append(SpectralNorm(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                  dilation=dilation, bias=use_bias), use_sn))
    norm_fun = get_norm_fun(norm_fun)
    main.append(norm_fun(out_channels))
    main.append(get_act_fun(act_fun))
    main = nn.Sequential(*main)
    return main


def dis_pred_conv_block(in_channels, out_channels, kernel_size, stride, padding, dilation, use_bias, type):
    padding = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
    main = []
    main.append(nn.ReflectionPad2d(padding))
    main.append(
        nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=0,
                  dilation=dilation, bias=use_bias))
    if type in ['ls', 'rals']:
        main.append(nn.Sigmoid())
    elif type in ['hinge', 'rahinge']:
        main.append(nn.Tanh())
    else:
        raise NotImplementedError("Adversarial loss [{}] is not found".format(type))
    main = nn.Sequential(*main)
    return main



def SpectralNorm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)
    return module



class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode, align_corners):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        out = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return out



def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.data.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.s = nn.Sigmoid()

    def forward(self, x):
        return x * self.s(x)


def get_act_fun(act_fun_type='LeakyReLU'):
    if isinstance(act_fun_type, str):
        if act_fun_type == 'LeakyReLU':
            return nn.LeakyReLU(0.2, inplace=True)
        elif act_fun_type == 'ReLU':
            return nn.ReLU(inplace=True)
        elif act_fun_type == 'Swish':
            return Swish()
        elif act_fun_type == 'SELU':
            return nn.SELU(inplace=True)
        elif act_fun_type == 'none':
            return nn.Sequential()
        else:
            raise NotImplementedError('activation function [%s] is not found' % act_fun_type)
    else:
        return act_fun_type()


class Identity(nn.Module):
    def forward(self, x):
        return x



def get_norm_fun(norm_fun_type='none'):
    if norm_fun_type == 'BatchNorm':
        norm_fun = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'InstanceNorm':
        norm_fun = functools.partial(nn.InstanceNorm2d, affine=True, track_running_stats=True)
    elif norm_fun_type == 'ILN':
        norm_fun = functools.partial(ILN)
    elif norm_fun_type == 'none':
        norm_fun = lambda x: Identity()
    else:
        raise NotImplementedError('normalization function [%s] is not found' % norm_fun_type)
    return norm_fun


class adaILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out


class Bottleneck(nn.Module):
    def __init__(self, dim, use_bias):
        super(Bottleneck, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = nn.InstanceNorm2d(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = nn.InstanceNorm2d(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out)

        return out + x

class AdaILNBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(AdaILNBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = adaILN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x, out

class ILN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (
                1 - self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)

        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):
        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w
