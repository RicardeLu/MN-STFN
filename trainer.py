import torch
from torch import nn
from tqdm import tqdm

import logging
from pathlib import Path
import time


class ExpRecorder():
    def __init__(self,save_path:str,model_name:str,model_file_name:str) -> None:
        """
        save_path  
            ./xxxx/xxxx/ 
                a dir str
        model_name
            model name 
                str
        model_file_name
            (model param)&(log file) save name  
                a file str

        """
        self.save_path = Path(save_path)
        self.model_name = model_name
        self.logger = None
        self._model_save_name = model_file_name
        self._init_save_path()
        self._init_logger()

    def _init_save_path(self) -> None:
        cur_time = time.strftime("%Y%m%d_%H%M%S",time.gmtime())
        self.save_path = self.save_path / f"{cur_time}-{self.model_name}"
        if not self.save_path.exists():
            self.save_path.mkdir()

    def _init_logger(self) -> None:
        self.logger = logging.getLogger()
        for handler in self.logger.handlers[0:]:
            self.logger.removeHandler(handler)
        self.logger.setLevel(level=logging.INFO)
        log_path = self.save_path / f"{self._model_save_name}.log"
        handler = logging.FileHandler(log_path, encoding='UTF-8')
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        self.logger.addHandler(handler)
        self.logger.addHandler(console)
        handler.close()
        console.close()
    def record(self,message:str) -> None:
        self.logger.info(message)

    @property
    def model_file_name(self) -> Path:
        return self.save_path / f"{self._model_save_name}.pt"





class RMSE(nn.Module):
    def __init__(self,scale=1) -> None:
        super(RMSE,self).__init__()
        self.scale = scale
        self.mse = torch.nn.MSELoss(reduction='mean')
    def forward(self,y,y_hat):
        y,y_hat = y *self.scale,y_hat*self.scale
        # mse = torch.mean(torch.pow(y-y_hat,2))
        mse_l = self.mse(y,y_hat)
        rmse = torch.sqrt(mse_l)
        return rmse

class MAPE(nn.Module):
    def __init__(self,scale=1) -> None:
        super(MAPE,self).__init__()
        self.scale = scale
    def forward(self,y,y_hat):
        y,y_hat = y *self.scale,y_hat*self.scale
        mask = y != 0
        y = y[mask]
        y_hat = y_hat[mask]
        diff = torch.abs(y-y_hat)
        percent_diff = diff/y
        percent_diff = percent_diff.clamp(min=1e-8)
        return torch.mean(percent_diff)
    

class LossFun(nn.Module):
    def __init__(self,scale):
        super(LossFun,self).__init__()
        self.RMSE = RMSE(scale)
        # self.MAPE = MAPE(scale,gamma)
        self.MAPE = MAPE(scale)
    def forward(self,y,y_hat):
        rmse_loss = self.RMSE(y,y_hat)
        mape_loss = self.MAPE(y,y_hat)
        loss = mape_loss + rmse_loss
        return loss,rmse_loss,mape_loss
    
def train_net(net:nn.Module|object,train_iter,test_iter,num_epochs,lr,scale,recoder:ExpRecorder,start_epoch=1):
    recoder.record("\ntrain on")
    optimizer = torch.optim.Adam(net.parameters(),lr=lr)
    Exp_opt = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.995)
    loss = LossFun(scale)
    min_test_mape = 100000
    min_test_rmse = 100000
    train_rmse_loss = []
    train_mape_loss = []
    test_rmse_loss = []
    test_mape_loss = []
    for epoch in range(start_epoch,num_epochs+1):
        net.train()
        for i,(X,y) in enumerate(tqdm(train_iter,leave=False,desc='epoch - %d'%(epoch))):
            optimizer.zero_grad()
            X = X.cuda()
            y = y.cuda()
            y_hat = net(X)
            l,lr,lm = loss(y,y_hat)
            l.backward()
            train_rmse_loss.append(lr)
            train_mape_loss.append(lm)
            optimizer.step() 
        net.eval()   
        with torch.no_grad():
            for i,(X,y) in enumerate(test_iter):
                X = X.cuda()
                y = y.cuda()
                y_hat = net(X)
                l,lr,lm = loss(y,y_hat)
                test_rmse_loss.append(lr)
                test_mape_loss.append(lm)
        mean_train_rmse_loss = float(sum(train_rmse_loss)/len(train_rmse_loss))
        mean_train_mape_loss = float(sum(train_mape_loss)/len(train_mape_loss))
        mean_test_rmse_loss =float(sum(test_rmse_loss)/len(test_rmse_loss))
        mean_test_mape_loss = float(sum(test_mape_loss)/len(test_mape_loss))
        if mean_test_mape_loss + mean_test_rmse_loss < min_test_rmse+min_test_mape:
            torch.save(net.state_dict(),recoder.model_file_name)
            recoder.record("save pt in epoch - %d"%(epoch))
            min_test_rmse = mean_test_rmse_loss
            min_test_mape  = mean_test_mape_loss 
        # if True:
        recoder.record("epoch - %d"%(epoch))
        recoder.record("train mean rmse loss:%f     train mean mape loss:%f\n"%(mean_train_rmse_loss,mean_train_mape_loss))
        recoder.record("test mean rmse loss:%f     test mean mape loss:%f\n"%(mean_test_rmse_loss,mean_test_mape_loss))
        recoder.record("min test rmse:%f   min test mape:%f\n"%(min_test_rmse,min_test_mape))
        train_mape_loss.clear() 
        train_rmse_loss.clear()
        test_rmse_loss.clear()
        test_mape_loss.clear()
        Exp_opt.step()


