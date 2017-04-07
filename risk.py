import  pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

class Risk:
    def __init__(self):
        self.train_filePath='/data/risk/PPD-First-Round-Data-Updated/PPD-First-Round-Data-Update/Training Set/'
        self.test_filePath='/data/risk/PPD-First-Round-Data-Updated/PPD-First-Round-Data-Update/Test Set/'
        
        self.master_train_file=self.train_filePath+'PPD_Training_Master_GBK_3_1_Training_Set.csv'
        self.LogInfo_train_file=self.train_filePath+'PPD_LogInfo_3_1_Training_Set.csv'
        self.Userupdate_train_file=self.train_filePath+'PPD_Userupdate_Info_3_1_Training_Set.csv'
        
        self.master_test_file=self.test_filePath+'PPD_Master_GBK_2_Test_Set.csv'         
        self.LogInfo_test_file=self.test_filePath+'PPD_LogInfo_2_Test_Set.csv'           
        self.Userupdate_test_file=self.test_filePath+'PPD_Userupdate_Info_2_Test_Set.csv'
        
        
        self.LogInfo_train=pd.read_csv(self.LogInfo_train_file,sep=',')
        self.Userupdate_train=pd.read_csv(self.Userupdate_train_file,sep=',')
        
        self.master_test=pd.read_csv(self.master_test_file,sep=',')         
        self.LogInfo_test=pd.read_csv(self.LogInfo_test_file,sep=',')           
        self.Userupdate_test=pd.read_csv(self.Userupdate_test_file,sep=',')
        
        self.fieldtype=pd.read_csv(r'/data/risk/field_type.csv',sep=',')

    def handle_data_nan(self):
        master_train=pd.read_csv(self.master_train_file,encoding='GBK',sep=',')
        for x,y in zip(self.fieldtype.field_name,self.fieldtype.field_type):
	        nan_rate = master_train[x].count()/float(master_train[x].size)
	        if nan_rate < 0.7:
		        master_train = master_train.drop([x],axis=1)
            	elif y == 'Categorical':
		        if master_train[x].dtypes == np.int64:
		            master_train[x][ pd.isnull(master_train[x]) ] = 99999
		        elif master_train[x].dtypes == np.float64:
		            master_train[x][ pd.isnull(master_train[x]) ] = 99999 
		        elif master_train[x].dtypes == np.object:
		            master_train[x][ pd.isnull(master_train[x]) ] = '99999'
		elif y == 'Numerical':
			if master_train[x].dtypes == np.int64:
			    master_train[x][ pd.isnull(master_train[x]) ] = master_train[x].mean()
			elif master_train[x].dtypes == np.float64:
			    master_train[x][ pd.isnull(master_train[x]) ] = master_train[x].mean()
       
        list_info_time = pd.to_datetime(master_train.ListingInfo)
        master_train['year'] = list_info_time.apply(lambda x : x.year)
        master_train['month'] = list_info_time.apply(lambda x : x.month)
        master_train['day'] = list_info_time.apply(lambda x : x.day)
        master_train = master_train.drop(['ListingInfo'],axis=1)
        master_train['year'] = master_train['year'].astype('category') 
        master_train['month'] = master_train['month'].astype('category') 
        master_train['day'] = master_train['day'].astype('category')         

 
        master_train.to_csv('/data/risk/master_nan.csv',encoding='GBK')          
        return master_train
   
    def handle_data_pre(self,master_train):
        le = LabelEncoder()
        for x in master_train.columns:
            if fieldtype[fieldtype.field_name == x].field_type == 'Categorical':
                le.fit(master_train[x].values)
                master_train[x] = le.transform(master_train[x])

         
 
    def train(master_train):
        X_train = master_train[ [x for x in master_train.columns if x!='target' ]   ] 
        Y_train = master_train[ ['target' ]   ]	
        	



	
if __name__ == '__main__':
    risk = Risk()
    risk.handle_data()
