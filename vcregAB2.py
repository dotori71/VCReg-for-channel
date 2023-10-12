import argparse
import torch
import torch.nn.functional as F
from torch import nn
import math

class VCReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def forward(self, y):
        # change shape [b, c, h, w]  to [b, c, hw]
        y = y.view(y.size(0), y.size(1), -1)

        std_loss=0
        cov_loss=0

        # ε=0.0001  γ=1
        if self.args.std_use:
            y_reshape= y.view(y.size(1),y.size(0), y.size(2)) # change shape [b, c, hw]  to [c, b, hw ]

            h_yj_sum=0
            for j in range(y.size(1)):
                std_yj = torch.sqrt(y_reshape[j].var(dim=0)+ 0.0001)   
                h_yj_sum = h_yj_sum + torch.mean(F.relu(1 - std_yj))   #hinge func has hw values, need to get mean 
            std_loss = h_yj_sum / y.size(1)
        print(self.args.cov_use)

        if self.args.cov_use:
        #if self.args.cov_method=="A":
            y_mean= self.get_batch_mean_y(y,self.args.batch_size) # y bar
            cov_y = self.get_cov_matrix_y(y,y_mean,self.args.batch_size, y.size(2))
            cov_loss = self.off_diagonal(cov_y).pow_(2).sum().div(y.size(1))  
            print(cov_loss)  
        #elif self.args.cov_method=="B":
            c_yi_sum=0
            hw=y.size(2)
            y_mean= self.get_batch_mean_y(y,self.args.batch_size) # y bar [c, hw]
            for ii in range(self.args.batch_size):
                C_yii=self.one_y_cov_matrix(y[ii],y_mean)
                c_yii=self.one_y_cov(C_yii,hw)
                c_yi_sum=c_yi_sum+c_yii
            cov_loss=c_yi_sum/self.args.batch_size
            print("1:",cov_loss)
            c_yi_sum=0
            hw=y.size(2)
            y_mean= self.get_batch_mean_y(y,self.args.batch_size) # y bar [c, hw]
            for ii in range(self.args.batch_size):
                C_yii=self.one_y_cov_matrix2(y[ii],y_mean)
                c_yii = self.one_y_cov(C_yii,hw)
                c_yi_sum=c_yi_sum+c_yii
            cov_loss=c_yi_sum/self.args.batch_size
            print("2:",cov_loss)

                
        loss = self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss    
        print(loss)
        return loss

    def one_y_cov_matrix(self,yi,y_mean):
        ch=yi.size(0)
        hw=yi.size(1)
        pair_num=math.comb(ch, 2)
        Cyi_sum=0
        for i in range(ch):
            for j in range(i + 1, ch):
                y1_copy=(yi[i]-y_mean[i]).reshape(hw,1)
                y2_copy=(yi[j]-y_mean[j]).reshape(1,hw)
                Cyi_sum=Cyi_sum+(torch.matmul(y1_copy, y2_copy))
        Cyi=Cyi_sum/hw/pair_num      
        return Cyi  
    
    def one_y_cov(self,C_yi,hw):
        cyi=C_yi.flatten().pow_(2).sum().div(hw)
        return cyi

    def get_batch_mean_y(self,y,batch_size):
        yi_sum=0
        for i in range(batch_size):
            yi_sum=yi_sum+y[i]
        y_mean=yi_sum/batch_size
        return y_mean
    def one_y_cov_matrix2(self,yi,y_mean):
        ch=yi.size(0)
        hw=yi.size(1)
        Cyi_sum=0
        for i in range(ch):
            for j in range(ch):
                if i==j:
                    continue
                else:
                    y1_copy=(yi[i]-y_mean[i]).reshape(hw,1)
                    y2_copy=(yi[j]-y_mean[j]).reshape(1,hw)
                    Cyi_sum=Cyi_sum+(torch.matmul(y1_copy, y2_copy))
        Cyi=Cyi_sum/hw/(ch-1)    
        return Cyi  
    def get_cov_matrix_y(self,y,y_mean,batch_size,hw):
        cov_sum=0
        for i in range(batch_size):
            cov_sum=cov_sum+((y[i]-y_mean)@((y[i]-y_mean).T))
        cov_y=cov_sum/hw/(batch_size-1)
        return cov_y

    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)
    # input shape
    parser.add_argument("--input_shape", type=str, default=[20,16, 16],
                        help='after encoding, the shape of each yi ')
    parser.add_argument("--batch-size", type=int, default=16,
                        help='batch size')

    # Loss
    parser.add_argument("--std_coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov_coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--std_use", action="store_true", help="use variance term or not")# store_true -> false
    parser.add_argument("--cov_use", action="store_false", help="use covariance term or not")# store_false -> true
    parser.add_argument("--cov_method",type=str,default="B",choices=["A", "B"],help="Covariance method (choose between 'A' and 'B')")
    #A: different channel same position
    #B: different channel diff position
    return parser


def main(args):
    #Namespace(input_shape=[3, 224, 224], batch_size=16, std_coeff=25.0, cov_coeff=1.0, std_use=False, cov_use=True)
    # Create a random tensor with the specified shape and batch size
    y = torch.randn((args.batch_size, *args.input_shape))
    model = VCReg(args)
    loss = model.forward(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VCReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
