import  pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

from xgboost.core import DMatrix

class Risk:
    def __init__(self):
        self.train_filePath='/data/risk/PPD-First-Round-Data-Updated/Training Set/'
        self.test_filePath='/data/risk/PPD-First-Round-Data-Updated/Test Set/'
        
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
        x=None
        y=None
        try:
            for x,y in zip(self.fieldtype.field_name,self.fieldtype.field_type):
                nan_rate = master_train[x].count()/float(master_train[x].size)
                if nan_rate < 0.7:
                    master_train = master_train.drop([x],axis=1)
                elif y == 'Categorical':
                    if master_train[x].dtypes == np.int64:
                        master_train[x].loc[ pd.isnull(master_train[x]) ] = 99999
                    elif master_train[x].dtypes == np.float64:
                        master_train[x].loc[ pd.isnull(master_train[x]) ] = 99999 
                    elif master_train[x].dtypes == np.object:
                        master_train[x].loc[ pd.isnull(master_train[x]) ] = '99999'
                elif y == 'Numerical':
                    if master_train[x].dtypes == np.int64:
                        master_train[x].loc[ pd.isnull(master_train[x]) ] = master_train[x].mean()
                    elif master_train[x].dtypes == np.float64:
                        master_train[x].loc[ pd.isnull(master_train[x]) ] = master_train[x].mean()
       
                 

        except Exception,e:
            print e
            print x
        master_train.to_csv('/data/risk/master_nan.csv',encoding='GBK')          
        return master_train
   
    def handle_data_pre(self,master_train):
        le = LabelEncoder()
        x=None
        try:
            for x in master_train.columns:
                if self.fieldtype[self.fieldtype.field_name == x].field_type.get_values()[0] == 'Categorical':
                    le.fit(master_train[x].values)
                    master_train[x] = le.transform(master_train[x])
            
            list_info_time = pd.to_datetime(master_train.ListingInfo)
            master_train['year'] = list_info_time.apply(lambda x : x.year)
            master_train['month'] = list_info_time.apply(lambda x : x.month)
            master_train['day'] = list_info_time.apply(lambda x : x.day)
            master_train = master_train.drop(['ListingInfo'],axis=1)
            #master_train['year'] = master_train['year'].astype('category') 
            #master_train['month'] = master_train['month'].astype('category') 
            #master_train['day'] = master_train['day'].astype('category')
        except Exception,e:
            print e
            print x
        return master_train
    
    def xgb_train(self,master_train):
        #X_train = master_train[ [x for x in master_train.columns if x!='target' ]   ] 
        #Y_train = master_train[ ['target' ]   ]
        
        xgb1 = xgb.XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=32, scale_pos_weight=1, seed=27)
            
        predictors = [x for x in master_train.columns if x!='target' ]
        target = 'target'
        self.modelfit(xgb1,master_train,target,predictors)
        
    
    def modelfit(self,alg, dtrain, target ,predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        try:
            if useTrainCV:
                xgb_param = alg.get_xgb_params()
                xgtrain = xgb.DMatrix(dtrain[predictors].values, label=dtrain[target].values)
                cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                    metrics='auc', early_stopping_rounds=early_stopping_rounds)
                alg.set_params(n_estimators=cvresult.shape[0])
            
            #Fit the algorithm on the data
            alg.fit(dtrain[predictors], dtrain[target],eval_metric='auc')
            
            #Predict training set:
            dtrain_predictions = alg.predict(dtrain[predictors])
            dtrain_predprob = alg.predict_proba(dtrain[predictors])[:,1]
        
            #Print model report:
            print "\nModel Report"
            print "Accuracy : %.4g" % metrics.accuracy_score(dtrain[target].values, dtrain_predictions)
            print "AUC Score (Train): %f" % metrics.roc_auc_score(dtrain[target], dtrain_predprob)
            
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')    
        except Exception,e:
            print e
if __name__ == '__main__':
    risk = Risk()
    master_train = risk.handle_data_nan()
    master_train = risk.handle_data_pre(master_train)
    risk.xgb_train(master_train)
    
    
    
    
    
    
