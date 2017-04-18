# -*- coding: utf-8 -*-
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
        self.train_filePath='/data/risk/PPD-First-Round-Data-Update/Training_Set/'
        self.test_filePath='/data/risk/PPD-First-Round-Data-Update/Test_Set/'
        
        self.master_train_file=self.train_filePath+'PPD_Training_Master_GBK_3_1_Training_Set.csv'
        self.LogInfo_train_file=self.train_filePath+'PPD_LogInfo_3_1_Training_Set.csv'
        self.Userupdate_train_file=self.train_filePath+'PPD_Userupdate_Info_3_1_Training_Set.csv'
        
        self.master_test_file=self.test_filePath+'PPD_Master_GBK_2_Test_Set.csv'         
        self.LogInfo_test_file=self.test_filePath+'PPD_LogInfo_2_Test_Set.csv'           
        self.Userupdate_test_file=self.test_filePath+'PPD_Userupdate_Info_2_Test_Set.csv'
        
        
        #self.master_train=pd.read_csv(self.master_train_file,sep=',')
        self.LogInfo_train=pd.read_csv(self.LogInfo_train_file,sep=',')
        self.Userupdate_train=pd.read_csv(self.Userupdate_train_file,sep=',')
       
       
        #self.master_test=pd.read_csv(self.master_test_file,sep=',')         
        self.LogInfo_test=pd.read_csv(self.LogInfo_test_file,sep=',')           
        self.Userupdate_test=pd.read_csv(self.Userupdate_test_file,sep=',')
        
        self.fieldtype=pd.read_csv(r'/data/risk/field_type.csv',sep=',')
        
        self.category_fields = self.fieldtype[self.fieldtype.field_type == 'Categorical'].field_name.values
        self.numeric_fields = self.fieldtype[self.fieldtype.field_type == 'Numerical'].field_name.values
        self.chinese_content_fields = ['UserInfo_2','UserInfo_4','UserInfo_7','UserInfo_8','UserInfo_19','UserInfo_20']
        
        
    def handle_nan_field(self,required_handle_nan_df):
        
        nan_field_count_series = required_handle_nan_df.isnull().sum().sort_values(ascending=False)/float(len(required_handle_nan_df)) 
        required_handle_nan_df.drop(nan_field_count_series[nan_field_count_series>0.9].index.values,inplace=True,axis=1)
        
        
        
        for x in required_handle_nan_df.columns.values:
            if x in self.category_fields:
                if x in self.chinese_content_fields:
                    required_handle_nan_df[x].loc[pd.isnull(required_handle_nan_df[x])] = (u'不详').encode('gbk')
                else:
                    if required_handle_nan_df[x].dtypes == np.int64:
                        required_handle_nan_df[x].loc[ pd.isnull(required_handle_nan_df[x]) ] = 99999
                    elif required_handle_nan_df[x].dtypes == np.float64:
                        required_handle_nan_df[x].loc[ pd.isnull(required_handle_nan_df[x]) ] = 99999.0 
                    elif required_handle_nan_df[x].dtypes == np.object:
                        required_handle_nan_df[x].loc[ pd.isnull(required_handle_nan_df[x]) ] = 'unknow'
            elif  x in self.numeric_fields:
                if required_handle_nan_df[x].dtypes == np.int64:
                        required_handle_nan_df[x].loc[ pd.isnull(required_handle_nan_df[x]) ] = required_handle_nan_df[x].mean()
                elif required_handle_nan_df[x].dtypes == np.float64:
                    required_handle_nan_df[x].loc[ pd.isnull(required_handle_nan_df[x]) ] = required_handle_nan_df[x].mean()
        
    def handle_little_variance(self,required_handle_variance_df):
        variance_series = required_handle_variance_df[[ x for x in  self.numeric_fields if x in required_handle_variance_df.columns]].std()
        required_handle_variance_df.drop(variance_series[variance_series < 0.1].index.values,inplace=True,axis=1)

    def handle_data_nan(self,data_file_path):
        
        data_df = pd.read_csv(data_file_path,sep=',')
       
        
        
        x=None
        y=None
        try:
            for x,y in zip(self.fieldtype.field_name,self.fieldtype.field_type):
                if  x not in data_df.columns:
                    continue
                nan_rate = data_df[x].count()/float(data_df[x].size)
                if nan_rate < 0.7:
                    data_df = data_df.drop([x],axis=1)
                elif y == 'Categorical':
                    if data_df[x].dtypes == np.int64:
                        data_df[x].loc[ pd.isnull(data_df[x]) ] = 99999
                    elif data_df[x].dtypes == np.float64:
                        data_df[x].loc[ pd.isnull(data_df[x]) ] = 99999 
                    elif data_df[x].dtypes == np.object:
                        data_df[x].loc[ pd.isnull(data_df[x]) ] = 'unknow'
                elif y == 'Numerical':
                    if data_df[x].dtypes == np.int64:
                        data_df[x].loc[ pd.isnull(data_df[x]) ] = data_df[x].mean()
                    elif data_df[x].dtypes == np.float64:
                        data_df[x].loc[ pd.isnull(data_df[x]) ] = data_df[x].mean()

        except Exception,e:
            print e
            print x
        #data_df.to_csv('/data/risk/master_train_test_nan.csv')          
        return data_df
   
    def handle_data_pre(self,master_train_test):
        le = LabelEncoder()
        #x=None
        try:
            
            list_info_time = pd.to_datetime(master_train_test.ListingInfo)
            master_train_test['year'] = list_info_time.apply(lambda x : x.year)
            master_train_test['month'] = list_info_time.apply(lambda x : x.month)
            master_train_test['day'] = list_info_time.apply(lambda x : x.day)
            master_train_test = master_train_test.drop(['ListingInfo'],axis=1)
            
        except Exception,e:
            print e
            
        return master_train_test
    
    def handle_data_cat_encode(self,master_train_test):
        
        
        le = LabelEncoder()
        try:
            for x in master_train_test.columns:
                if x not in self.fieldtype.field_name.values:
                    continue
                
                if self.fieldtype[self.fieldtype.field_name == x].field_type.get_values()[0] == 'Categorical':
                    le.fit(master_train_test[x].values)
                    master_train_test[x] = le.transform(master_train_test[x])
        
        except Exception,e:
            print e
            print x
        return master_train_test
    
    #calc target sum for group by category field
    def handle_category_count(self,master_train_test):
        le = LabelEncoder()
        cat_cols = [x for x in risk.fieldtype[ risk.fieldtype.field_type == 'Categorical'  ].field_name  if x in master_train.columns ]
        for cat in cat_cols:
            if master_train[cat].dtype == 'object':
                master_train[cat] = le.fit_transform( list( map( lambda x:x.decode('gbk'),master_train[cat].values) ) )
            
            
            
            master_train['key_1'] =  master_train[cat].astype('category').values.codes
            grp = master_train.groupby(['key_1'])
              
            
            target0 = master_train[cat][master_train.target == 0].value_counts()
            target1 = master_train[cat][master_train.target == 1].value_counts()
            df = pd.DataFrame({cat+'_wy':target1,cat+'_w_wy':target0})
            df.fillna(0,inplace=True)
            master_train[cat+'_exp'] = df[cat+'_wy'].astype('float')/(df[cat+'_wy'].astype('float') + df[cat+'_w_wy'].astype('float'))
    
    
    
    def xgb_train(self,master_train_test):
        #X_train = master_train[ [x for x in master_train.columns if x!='target' ]   ] 
        #Y_train = master_train[ ['target' ]   ]
        
        xgb1 = xgb.XGBClassifier( learning_rate =0.1, n_estimators=1000, max_depth=5, min_child_weight=1, gamma=0, subsample=0.8, 
                                  colsample_bytree=0.8, objective= 'binary:logistic', nthread=32, scale_pos_weight=1, seed=27)
            
        predictors = [x for x in master_train.columns if x not in ['Idx','target','source'] ]
        target = 'target'
        self.modelfit(xgb1,master_train_test,target,predictors)
        
    
    def modelfit(self,alg, dtrain, target ,predictors,useTrainCV=True, cv_folds=5, early_stopping_rounds=50):
        try:
            
            train_set  = dtrain[dtrain.source == 'train']
            test_set = dtrain[dtrain.source == 'test']
            
            if useTrainCV:
                xgb_param = alg.get_xgb_params()
                xgtrain = xgb.DMatrix(train_set[predictors].values, label=train_set[target].values)
                cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
                    metrics='auc', early_stopping_rounds=early_stopping_rounds)
                alg.set_params(n_estimators=cvresult.shape[0])
            
            #Fit the algorithm on the data
            alg.fit(train_set[predictors], train_set[target],eval_metric='auc')
            
            #Predict training set:
            dtrain_predictions = alg.predict(test_set[predictors])
            dtrain_predprob = alg.predict_proba(test_set[predictors])[:,1]
        
            #Print model report:
            print "\nModel Report"
            print "Accuracy : %.4g" % metrics.accuracy_score(test_set[target].values, dtrain_predictions)
            print "AUC Score (Train): %f" % metrics.roc_auc_score(test_set[target].values, dtrain_predprob)
            
            feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
            feat_imp.plot(kind='bar', title='Feature Importances')
            plt.ylabel('Feature Importance Score')    
        except Exception,e:
            print e

if __name__ == '__main__':
    risk = Risk()
#     master_train = risk.handle_data_nan(risk.master_train_file)
#     master_test = risk.handle_data_nan(risk.master_test_file)
#     
#     master_train = risk.handle_data_pre(master_train)
#     risk.handle_category_count(master_train)
#     
#     master_test = risk.handle_data_pre(master_test)
   
    master_train = pd.read_csv(risk.master_train_file)
    risk.handle_nan_field(master_train)
    risk.handle_little_variance(master_train)
    print master_train.shape
    
#     master_test['target'] = 0
#     master_test['source'] = 'test'
#     master_train['source'] = 'train'
#     
#     master_train_test= pd.concat([master_train,master_test],ignore_index=True)
#     
#     master_train_test.UserInfo_24[38293] = 'unknow'   #特殊处理，仅此处的值无法进行decode（‘gbk’）编码
#     
#     #for x in [x for x in master_train_test.columns if   master_train_test[x].dtype == 'object' and  x.find('UserInfo_24')<0 ]:  #做了166行处理后，不用排除该列了
#     
#     #将object类型得列都进行decode操作，不在单选包含中文得列了
#     for x in [x for x in master_train_test.columns if   master_train_test[x].dtype == 'object' ]:               
#         master_train_test[x] =  master_train_test[x].apply(lambda x:x.decode('gbk'))
#     
#     
# #     master_train_test.UserInfo_2 = master_train_test.UserInfo_2.apply(lambda x:x.decode('gbk') )
# #     master_train_test.UserInfo_4 = master_train_test.UserInfo_4.apply(lambda x:x.decode('gbk') )
# #     master_train_test.UserInfo_7 = master_train_test.UserInfo_7.apply(lambda x:x.decode('gbk') )
# #     master_train_test.UserInfo_8 = master_train_test.UserInfo_8.apply(lambda x:x.decode('gbk') )
# #     master_train_test.UserInfo_9 = master_train_test.UserInfo_9.apply(lambda x:x.decode('gbk') )
#     
#     master_train_test = risk.handle_data_cat_encode(master_train_test)
#     
#     #master_train_test = master_train_test.drop(['Idx'],axis=1)
# 
#     #master_train = risk.handle_data_pre(master_train)
#     risk.xgb_train(master_train_test)
    
    
    
    
    
    
