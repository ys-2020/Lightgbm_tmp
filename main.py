import processing
from lightgbm import LGBMClassifier
from lightgbm import plot_importance as LGim
from xgboost import XGBClassifier
from xgboost import plot_importance as XGim
from sklearn.metrics import accuracy_score
import time
import joblib
import torch
import os



path=r'C:\Users\HW\Desktop'  ##填写数据集路径，填写到具体文件夹名，如左

training_method=1;             ##选择模型，1代表LGBM，2代表XGboost

columnNames1,rows1 = processing.load_data(path+r'\train.csv')  ##填写具体文件名
columnNames2,rows2 = processing.load_data(path+r'\test.csv')
columnNames3,rows3 = processing.load_data(path+r'\data.csv')

columnNames1,x_train,y_train=processing.separate(columnNames1, rows1,'tag')
columnNames2,x_test,y_test=processing.separate(columnNames2, rows2,'tag')
columnNames3,x_data,y_data=processing.separate(columnNames3, rows3,'maxval')


if(training_method==1):
   model = LGBMClassifier(n_estimators=1000) #树的数量
else:
   model = XGBClassifier(use_label_encoder=False)


#feature_names = ['nnz', 'mat_size', 'dev_row','k', 'CUDA_cores', 'bandwidth', 'L2_cache']
#feature_names = ['nnz', 'mat_size', 'dev_row','k']
#train the model
#model.fit(x_train, y_train,feature_name = feature_names)
model.fit(x_train, y_train)

#save the model
joblib.dump(model, r'C:\Users\HW\Desktop\model.txt')
#model = joblib.load(r'C:\Users\HW\Desktop\model.txt')

time1=time.time()
# make predictions for test data
y_predict = model.predict(x_test)

time2=time.time()

print(time2-time1)
print('\n',"predict :",y_predict)#输出预测特征值
print(" real tag:",y_test)#输出真实特征值


#预测准确率评估
predictions = [round(value) for value in y_predict]
# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))


# if(training_method==1):   #显示各Feature的importance
#    LGim(model, grid=False,ignore_zero = False)
# else:
#    XGim(model,grid=False)
#
# from matplotlib import pyplot as plt
# plt.title('PR_CM(hardware_paras)')
# plt.yticks([6,5,4,3,2,1,0],['mat_size','nnz','dev_row','CUDA_cores','k', 'L2_cache','bandwidth'])
# plt.show()

os.system('pause')