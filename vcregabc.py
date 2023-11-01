import argparse
import torch
import torch.nn.functional as F
from torch import nn
import math
import time
class VCReg(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def forward(self, y):
        # change shape [b, c, h, w]  to [b, c, hw]
        y = y.view(y.size(0), y.size(1), -1)

        std_loss=0
        cov_loss=0

        if self.args.std_use:
            y_reshape= y.view(y.size(1),y.size(0), y.size(2)) # change shape [b, c, hw]  to [c, b, hw ]

            h_yj_sum=0
            for j in range(y.size(1)):
                std_yj = torch.sqrt(y_reshape[j].var(dim=0)+ 0.0001)   
                h_yj_sum = h_yj_sum + torch.mean(F.relu(1 - std_yj))   #hinge func has hw values, need to get mean 
            std_loss = h_yj_sum / y.size(1)
        print(self.args.cov_use)

        if self.args.cov_use:
            if self.args.cov_method=="A":
                if self.args.cov_methodA=="1":
                    y_mean= self.get_one_y_mean(y,self.args.batch_size) # y bar
                    cov_y = self.get_cov_matrix_y(y,y_mean,self.args.batch_size, y.size(2))
                    cov_loss = self.off_diagonal(cov_y).pow_(2).sum().div(y.size(1))  
                    print(cov_loss)  
                if self.args.cov_methodA=="2":
                    y_mean= self.get_batch_mean_y(y,self.args.batch_size) # y bar
                    cov_y = self.get_cov_matrix_y2(y,y_mean,self.args.batch_size, y.size(2))
                    cov_loss = self.off_diagonal(cov_y).pow_(2).sum().div(y.size(1))  
                    print(cov_loss)  
            elif self.args.cov_method=="B":
                c_yi_sum=0
                ch=y.size(1)
                hw=y.size(2)
                y_mean= self.get_batch_mean_y(y,self.args.batch_size) # y bar [c, hw]
                for ii in range(self.args.batch_size):
                    C_yii=self.one_y_cov_matrix(y[ii],y_mean)
                    c_yii = self.one_y_cov(C_yii,hw,ch)
                    c_yi_sum=c_yi_sum+c_yii
                cov_loss=c_yi_sum/self.args.batch_size
                print(cov_loss)
            elif self.args.cov_method=="C":
                # start_time = time.time()
                b=y.size(0)
                co_sum=0
                for ii in range(b):
                    cov_lossii=self.cov_c(y[ii])
                    co_sum=co_sum+cov_lossii
                cov_loss=co_sum/b/y.size(1)/(y.size(1)-1)/y.size(2)
                #print(cov_loss)
                # end_time = time.time()
                # runtime = end_time - start_time
                # print(f"1forRuntime: {runtime} seconds")
                
        loss = self.args.std_coeff * std_loss + self.args.cov_coeff * cov_loss    
        print(loss)
        return loss
    
    def rbf_kernel(self,x, y, gamma):
        distance_square = (x - y).pow_(2).sum(dim=1)  # Euclidean distance
        k=torch.exp(-gamma * distance_square)
        k=k.sum()-1
        return k

    def cov_c(self,y):
        ch=y.size(0)
        hw=y.size(1)
        cov_sum=0
        for i in range(ch):
            ci_kernels = self.rbf_kernel(y[i].unsqueeze(0), y, 0.01)
            cov_sum=cov_sum + ci_kernels
        return cov_sum 

    # def rbf_kernel(self,x, y, gamma):
    #     distance = torch.norm(x - y)  # Euclidean distance
    #     k=torch.exp(-gamma * distance**2)
    #     return k

    # def cov_c(self,y):
    #     ch=y.size(0)
    #     hw=y.size(1)
    #     cov_sum=0
    #     for i in range(ch):
    #         for j in range(i+1,ch):
    #             c_ch_pair=self.rbf_kernel(y[i],y[j],0.01)
    #             cov_sum=cov_sum+c_ch_pair
    #     return cov_sum*2  

    def get_one_y_mean(self,y,batch_size):
        average_channel = torch.mean(y[:, :, :], dim=2)
        y_mean = torch.unsqueeze(average_channel, dim=2).repeat(1, 1, y.size(2))
        return y_mean

    def one_y_cov(self,C_yi,hw,ch):
        total_sum = 0
        for i in range(ch):
                x1, y1 = i*hw, i*hw
                x2, y2 = (i+1)*hw-1, (i+1)*hw-1
                region_sum = C_yi[x1:x2, y1:y2].flatten().pow_(2).sum()
                total_sum += region_sum
        #all
        cyi_all=C_yi.flatten().pow_(2).sum()
        #same channel compare to same channel
        cyi_remove=total_sum.item()

        cyi=(cyi_all-cyi_remove).div(hw)
        return cyi

    def get_batch_mean_y(self,y,batch_size):
        yi_sum=0
        for i in range(batch_size):
            yi_sum=yi_sum+y[i]
        y_mean=yi_sum/batch_size
        return y_mean
    
    def one_y_cov_matrix(self,yi,y_mean):
        ch=yi.size(0)
        hw=yi.size(1)
        flattened = yi.view(-1)
        flattened_mean = y_mean.view(-1)
        yy = flattened - flattened_mean
        Cyi = torch.ger(yy, yy) #outer_product
        #flattend (ch*hw)(1d)
        Cyi=Cyi/hw/(ch-1)    
        return Cyi  
    
    def get_cov_matrix_y(self,y,y_mean,batch_size,hw):
        cov_sum=0
        for i in range(batch_size):
            cov_sum=cov_sum+((y[i]-y_mean[i])@((y[i]-y_mean[i]).T))
        cov_y=cov_sum/hw/(batch_size-1)
        return cov_y
    
    def get_cov_matrix_y2(self,y,y_mean,batch_size,hw):
        cov_sum=0
        for i in range(batch_size):
            cov_sum=cov_sum+((y[i]-y_mean)@((y[i]-y_mean).T))
        cov_y=cov_sum/(batch_size-1)
        return cov_y
    
    def off_diagonal(self,x):
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def get_arguments():
    parser = argparse.ArgumentParser(description="Pretrain a resnet model with VICReg", add_help=False)
    # input shape
    parser.add_argument("--input_shape", type=str, default=[4,2,2],
                        help='after encoding, the shape of each yi ')
    parser.add_argument("--batch-size", type=int, default=3,
                        help='batch size')

    # Loss
    parser.add_argument("--std_coeff", type=float, default=25.0,
                        help='Variance regularization loss coefficient')
    parser.add_argument("--cov_coeff", type=float, default=1.0,
                        help='Covariance regularization loss coefficient')
    parser.add_argument("--std_use", action="store_true", help="use variance term or not")# store_true -> false
    parser.add_argument("--cov_use", action="store_false", help="use covariance term or not")# store_false -> true
    parser.add_argument("--cov_method",type=str,default="C",choices=["A", "B","C"],help="Covariance method (choose between 'A' and 'B')")
    parser.add_argument("--cov_methodA",type=str,default="2",choices=["1", "2"],help="Covariance method (choose between '1' and '2')")
    #A: different channel same position(LINEAR)
    #B: different channel diff position(LINEAR)
    #C: different channel              (NON-LINEAR)
    #A1:     /HW & centered over one y
    #A2: w/o /HW & centered over batch y
    return parser


def main(args):
    #Namespace(input_shape=[320, 16, 16], batch_size=4, std_coeff=25.0, cov_coeff=1.0, std_use=False, cov_use=True)
    # Create a random tensor with the specified shape and batch size
    # int_tensor = torch.randint(low=0, high=255, size=(9,320, 16 ,16), dtype=torch.int32)
    # # y = torch.randn((args.batch_size, *args.input_shape))
    # torch.save(int_tensor, 'int2.pth')    
    file_path = 'int2.pth'
    y = torch.load(file_path)
    y = y.to(torch.float32)
    print(y)
    model = VCReg(args)
    loss = model.forward(y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser('VCReg training script', parents=[get_arguments()])
    args = parser.parse_args()
    main(args)
