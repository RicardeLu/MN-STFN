import h5py
import os
import numpy as np
import torch

from config import ENV_CON_B,ENV_CON_N
from datetime import datetime,timedelta
from torch.utils.data import Dataset,DataLoader

from pathlib import Path


# -------------- Taxi BJ ---------------

class Database_Manage_B():
    def __init__(self) -> None:
        #------
        self.all_flow_data = {}
        self.all_backgroud_data = {}

        self.flow_data = []
        self.files = []
        self.input_sequence_length = 0
        self.output_sequence_length = 0
        self.genarated_data = {}
        self.all_incomplet_day = []
        self.load_data()
        self.file_save_name = ""
        
        
    def load_data(self):
        print('load data')
        for file_name in ENV_CON_B['flow_files_name']:
            with h5py.File(ENV_CON_B['path'] + file_name,'r') as f:
                print('----------------------------------')
                print('read \nfile:%s \ndata number:%d'%(file_name,len(f['date'])))
                day_count = {}
                for dataItem in zip(f['date'],f['data']):
                    date = dataItem[0].decode()
                    if date[:8] not in day_count.keys():
                        day_count[date[:8]] = 1
                    else:
                        day_count[date[:8]] += 1
                    self.all_flow_data[date] = dataItem[1]
                incomplet_day = []
                for day in  day_count.keys():
                    if day_count[day] != 48:
                        incomplet_day.append(day)
                print('incomplet day:%s\n'%(str(incomplet_day)))
                self.all_incomplet_day += incomplet_day

    def genarate_data(self,input_sequence_length = 4,output_sequence_length = 5,file_save_name=" "):
        self.file_save_name = file_save_name
        self.genarated_data.clear()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        for date_key in self.all_flow_data.keys():
            one_sequence_data = self.create_one_sequence_data_without_incomplete(date_key)
            if one_sequence_data[0] == 1:
                self.genarated_data[date_key] = one_sequence_data[1]
        return self.genarated_data


    def create_one_sequence_data_without_incomplete(self,date_key:str)->tuple:
        if date_key[:8] in self.all_incomplet_day:
            return (0,0)
        input_sequence = [date_key]
        output_sequence = []
        for i in range(self.input_sequence_length-1):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key  in self.all_flow_data.keys():
                if next_closeness_date_key[:8] in self.all_incomplet_day:
                    return (0,0)
                input_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)  
        for i in range(self.output_sequence_length):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key in self.all_flow_data.keys():
                if next_closeness_date_key[:8] in self.all_incomplet_day:
                    return (0,0)
                output_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)
        return (1,[input_sequence,output_sequence])

    
    def get_next_closeness_data_key(self,date_key):
        time_interval = timedelta(minutes=30)
        return self.get_next_date_key(date_key,time_interval)     

    def get_last_day_date_key(self,date_key):
        time_interval = timedelta(days=1)
        return self.get_last_date_key(date_key,time_interval)   

    def get_last_week_date_key(self,date_key):
        time_interval = timedelta(weeks=1)
        return self.get_last_date_key(date_key,time_interval)   

    def get_next_date_key(self,date_key:str,time_interval:timedelta):
        date_key = date_key[0:8] + '%02d'%(  (int(date_key[8:])*30 - 30)//60    )      + '%02d'%((int(date_key[8:])*30 - 30)%60)
        date_key_time = datetime.strptime(date_key,'%Y%m%d%H%M')
        date_key_time += time_interval
        year = date_key_time.year
        month = date_key_time.month
        day = date_key_time.day
        fregment = (date_key_time.hour*60 + date_key_time.minute + 30)//30
        return '%d%02d%02d%02d'%(year,month,day,fregment)
    
    def get_last_date_key(self,date_key:str,time_interval:timedelta):
        date_key = date_key[0:8] + '%02d'%(  (int(date_key[8:])*30 - 30)//60    )      + '%02d'%((int(date_key[8:])*30 - 30)%60)
        date_key_time = datetime.strptime(date_key,'%Y%m%d%H%M')
        date_key_time -= time_interval
        year = date_key_time.year
        month = date_key_time.month
        day = date_key_time.day
        fregment = (date_key_time.hour*60 + date_key_time.minute + 30)//30
        return '%d%02d%02d%02d'%(year,month,day,fregment)
        
    def save_data(self):
        path:Path = Path(ENV_CON_B['path_p']) / ENV_CON_B['path_d'] /  ENV_CON_B['data_name']
        path.mkdir(parents=True,exist_ok=True)
        file_path =  path / f"{self.file_save_name}.h5"
        with h5py.File(file_path,'w') as f:
            print(path,len(self.genarated_data.keys()))
            save_data = [ str(x) for x in self.genarated_data.values()]
            f.create_dataset('all_seq_date_key',data=save_data)
            
   
class FlowData_B(Dataset):
    def __init__(self,forecast_step=1) -> None:
        super().__init__()
        self.all_flow_data = {}
        self.all_seq_key_data = []
        self.min = float("inf")
        self.max = float("-inf")
        self.forecast_step = forecast_step
        self.init_data()
    def init_data(self):
        for file_name in ENV_CON_B['flow_files_name']:
            with h5py.File(ENV_CON_B['path'] + file_name,'r') as f:
                for dataItem in zip(f['date'],f['data']):
                    self.all_flow_data[dataItem[0].decode()] = dataItem[1]
                    if dataItem[1].max() > self.max:
                        self.max = dataItem[1].max()
                    if dataItem[1].min() < self.min:
                        self.min = dataItem[1].min()
        print(self.min,self.max)
        path = Path(ENV_CON_B['path_p']) / ENV_CON_B['path_d'] / ENV_CON_B['data_name'] / f"flow_data_4_{self.forecast_step}.h5"
        with h5py.File(path,'r') as f:
            for key in f['all_seq_date_key']:
                self.all_seq_key_data.append(eval(key.decode()))
    def __getitem__(self, index):
        seq_key = self.all_seq_key_data[index]
        input__data = []
        output_data = []
        for input_key in seq_key[0]:
            input__data.append(self.transform(self.all_flow_data[input_key]))
        for output_key in seq_key[1]:
            output_data.append(self.transform(self.all_flow_data[output_key]))
        return (torch.Tensor(np.stack(input__data,axis=0)),torch.Tensor(np.stack(output_data,axis=0)))
    def transform(self,X):
        X = (X-self.min)/(self.max-self.min)
        return X
    def __len__(self):
        return len(self.all_seq_key_data)


# ------------- NYC Bike -------------


class Database_Manage_N():
    def __init__(self) -> None:
        #------
        self.all_flow_data = {}
        self.all_backgroud_data = {}

        self.flow_data = []
        self.files = []
        self.input_sequence_length = 0
        self.output_sequence_length = 0
        self.genarated_data = {}
        self.all_incomplet_day = []
        self.load_data()
        self.file_save_name = ""
        
        
    def load_data(self):
        print('load data')
        for file_name in ENV_CON_N['flow_files_name']:
            with h5py.File(ENV_CON_N['path'] + file_name,'r') as f:
                print('----------------------------------')
                print('read \nfile:%s \ndata number:%d'%(file_name,len(f['date'])))
                day_count = {}
                for dataItem in zip(f['date'],f['data']):
                    date = dataItem[0].decode()
                    if date[:8] not in day_count.keys():
                        day_count[date[:8]] = 1
                    else:
                        day_count[date[:8]] += 1
                    self.all_flow_data[date] = dataItem[1]
                incomplet_day = []
                for day in  day_count.keys():
                    if day_count[day] != 48:
                        incomplet_day.append(day)
                print('incomplet day:%s\n'%(str(incomplet_day)))
                self.all_incomplet_day += incomplet_day

    def genarate_data(self,input_sequence_length = 4,output_sequence_length = 5,file_save_name=" "):
        self.file_save_name = file_save_name
        self.genarated_data.clear()
        self.input_sequence_length = input_sequence_length
        self.output_sequence_length = output_sequence_length
        for date_key in self.all_flow_data.keys():
            one_sequence_data = self.create_one_sequence_data_without_env(date_key)
            if one_sequence_data[0] == 1:
                self.genarated_data[date_key] = one_sequence_data[1]
        return self.genarated_data


    def create_one_sequence_data_with_all_env(self,date_key:str)->tuple:
        def is_complete_date_key(date_key_c):
            if date_key_c[:8]  in self.all_incomplet_day:
                return False
            if date_key_c not in self.all_backgroud_data.keys():
                return False
            return True   
        if not is_complete_date_key(date_key):
            return (0,0)
        input_sequence = [date_key]
        output_sequence = []
        for i in range(self.input_sequence_length-1):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key  in self.all_flow_data.keys():
                if not is_complete_date_key(next_closeness_date_key):
                    return (0,0)
                input_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)  
        for i in range(self.output_sequence_length):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key in self.all_flow_data.keys():
                if not is_complete_date_key(next_closeness_date_key):
                    return (0,0)
                output_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)
        return (1,[input_sequence,output_sequence])
    

    def create_one_sequence_data_without_env(self,date_key:str)->tuple:
        input_sequence = [date_key]
        output_sequence = []
        for i in range(self.input_sequence_length-1):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key  in self.all_flow_data.keys():
                input_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)  
        for i in range(self.output_sequence_length):
            next_closeness_date_key = self.get_next_closeness_data_key(date_key)
            if next_closeness_date_key in self.all_flow_data.keys():
                output_sequence.append(next_closeness_date_key)
                date_key = next_closeness_date_key
            else:
                return (0,0)
        return (1,[input_sequence,output_sequence])
    
    def get_next_closeness_data_key(self,date_key):
        time_interval = timedelta(minutes=30)
        return self.get_next_date_key(date_key,time_interval)     

    def get_last_day_date_key(self,date_key):
        time_interval = timedelta(days=1)
        return self.get_last_date_key(date_key,time_interval)   

    def get_last_week_date_key(self,date_key):
        time_interval = timedelta(weeks=1)
        return self.get_last_date_key(date_key,time_interval)   

    def get_next_date_key(self,date_key:str,time_interval:timedelta):
        date_key = date_key[0:8] + '%02d'%(  (int(date_key[8:])*30 - 30)//60    )      + '%02d'%((int(date_key[8:])*30 - 30)%60)
        date_key_time = datetime.strptime(date_key,'%Y%m%d%H%M')
        date_key_time += time_interval
        year = date_key_time.year
        month = date_key_time.month
        day = date_key_time.day
        fregment = (date_key_time.hour*60 + date_key_time.minute + 30)//30
        return '%d%02d%02d%02d'%(year,month,day,fregment)
    
    def get_last_date_key(self,date_key:str,time_interval:timedelta):
        date_key = date_key[0:8] + '%02d'%(  (int(date_key[8:])*30 - 30)//60    )      + '%02d'%((int(date_key[8:])*30 - 30)%60)
        date_key_time = datetime.strptime(date_key,'%Y%m%d%H%M')
        date_key_time -= time_interval
        year = date_key_time.year
        month = date_key_time.month
        day = date_key_time.day
        fregment = (date_key_time.hour*60 + date_key_time.minute + 30)//30
        return '%d%02d%02d%02d'%(year,month,day,fregment)
        
    def save_data(self):
        path:Path = Path(ENV_CON_N['path_p']) / ENV_CON_N['path_d'] /  ENV_CON_N['data_name']
        path.mkdir(parents=True,exist_ok=True)
        file_path =  path / f"{self.file_save_name}.h5"
        with h5py.File(file_path,'w') as f:
            print(path,len(self.genarated_data.keys()))
            save_data = [ str(x) for x in self.genarated_data.values()]
            f.create_dataset('all_seq_date_key',data=save_data)


class FlowData_N(Dataset):
    def __init__(self,forecast_step=1) -> None:
        super().__init__()
        self.all_flow_data = {}
        self.all_seq_key_data = []
        self.min = float("inf")
        self.max = float("-inf")
        self.forecast_step = forecast_step
        self.init_data()
    def init_data(self):
        for file_name in ENV_CON_N['flow_files_name']:
            with h5py.File(ENV_CON_N['path'] + file_name,'r') as f:
                for dataItem in zip(f['date'],f['data']):
                    self.all_flow_data[dataItem[0].decode()] = dataItem[1]
                    if dataItem[1].max() > self.max:
                        self.max = dataItem[1].max()
                    if dataItem[1].min() < self.min:
                        self.min = dataItem[1].min()
        print(self.min,self.max)
        path = Path(ENV_CON_N['path_p']) / ENV_CON_N['path_d'] / ENV_CON_N['data_name'] / f"flow_data_4_{self.forecast_step}.h5"
        with h5py.File(path,'r') as f:
            for key in f['all_seq_date_key']:
                self.all_seq_key_data.append(eval(key.decode()))
    def __getitem__(self, index):
        seq_key = self.all_seq_key_data[index]
        input__data = []
        output_data = []
        for input_key in seq_key[0]:
            input__data.append(self.transform(self.all_flow_data[input_key]))
        for output_key in seq_key[1]:
            output_data.append(self.transform(self.all_flow_data[output_key]))
        return (torch.Tensor(np.stack(input__data,axis=0)),torch.Tensor(np.stack(output_data,axis=0)))
    def transform(self,X):
        X = (X-self.min)/(self.max-self.min)
        return X
    def __len__(self):
        return len(self.all_seq_key_data)






