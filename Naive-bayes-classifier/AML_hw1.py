#MNIST Image Classification
#Untouched: Do not re-center the digits, but use the images as is.
#Bounding box: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels is centered in the box.
#Stretched bounding box: Construct a 20 x 20 bounding box so that the horizontal (resp. vertical) range of ink pixels runs the full horizontal (resp. vertical) range of the box. Obtaining this representation will involve rescaling image pixels: you find the horizontal and vertical ink range, cut that out of the original image, then resize the result to 20 x 20. Once the image has been re-centered, you can compute features.


import numpy as np
import pandas as pd
import scipy.misc
from pandas import Series, DataFrame
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import string
flatten = lambda l: [item for sublist in l for item in sublist]

def two_pixels(m):
    m[m >= 50] = 255
    m[m < 50] = 0
    m[m >= 50] = 1
    return m

def bound_box(lines):
    # make lines only have 1 and 0
    #lines[lines >= 100] = 255
    #lines[lines < 100] = 0
    #lines[lines >= 100] = 1
    #find the boundary of 28*28 image array
    m = lines.values.reshape((28, 28))
    row_del=[]
    col_del=[]
    for j in range(28):
        if (len(m[j].nonzero()[0]) == 0):
            row_del.append(j)
        if (len(m[:,j].nonzero()[0]) == 0):
            col_del.append(j)
    #print(row_del,col_del)
    m = np.delete(m, row_del, 0)
    m = np.delete(m, col_del, 1)
    #resize to 20*20
    m=scipy.misc.imresize(m, (20,20))
    # make lines only have 1 and 0
    m=two_pixels(m)
    return m

def stretched_datafr(df):
    new_df = pd.DataFrame(columns=df.columns[0:400])
    indexlist = df.columns[0:400]
    for i in range(df.iloc[:, 0].size):
        line = df.loc[i][0:]
        new_array = bound_box(line)
        x = flatten(new_array.tolist())
        ser = Series(x, index=indexlist)
        new_df.loc[i] = ser
    return new_df

def digits_image(test_X,pred,dim,method_num):
    image_df= pd.DataFrame(test_X)
    pd.concat([image_df, pd.DataFrame(columns=["t_label"])])
    image_df["t_label"]=pred
    digits_df=image_df.groupby(image_df["t_label"]).mean()
    digits_df=digits_df.drop(columns=["t_label"])
    d_index = digits_df.index
    for i in d_index:
        line = digits_df.ix[i]
        m = line.values.reshape((dim, dim))
        mystr = "/Users/panzhang/Desktop/CS498AML/HW1/" + "_".join(["panz2", str(method_num), str(i)]) + ".png"
        plt.imsave(mystr, m, cmap=cm.gray)

def run_model(train_X,train_Y,val_X, val_Y,test_X,methodID,model):
    model.fit(train_X,train_Y)
    test_pred = model.predict(test_X)
    val_pred = model.predict(val_X)
    score = accuracy_score(val_Y,val_pred)
    print(score)
    test_pred = Series(test_pred)
    x1 = test_pred.index.values
    out = pd.concat([Series(x1), test_pred], axis=1)
    out.columns = ["ImageId", "Label"]
    mystr = "/Users/panzhang/Desktop/CS498AML/HW1/" + "_".join(["panz2", str(methodID)]) + ".csv"
    out.to_csv(mystr, index=False, header=True)
    return test_pred

#data processing
(('/Users/panzhang/Desktop/CS498AML/HW1/train.csv',sep=',')
train=train.drop(columns=["Unnamed: 0"])
train_label=train["label"]
train=train.drop(columns=["label"])
train=two_pixels(train)
untouched_train=pd.concat([train,train_label],axis=1)
untouched_train.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_train.csv',index=False,header=True)

stretched_train=stretched_datafr(train)
stretched_train=pd.concat([stretched_train,train_label],axis=1)
stretched_train.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_train.csv',index=False,header=True)

val = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/val.csv',sep=',')
val_label=val["label"]
val = val.drop(columns=["label"])
val=two_pixels(val)
untouched_val=pd.concat([val,val_label],axis=1)
untouched_val.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_val.csv',index=False,header=True)
stretched_val=stretched_datafr(val)
stretched_val=pd.concat([stretched_val,val_label],axis=1)
stretched_val.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_val.csv',index=False,header=True)

test = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/test.csv',sep=',',header=None)
untouched_test=two_pixels(test)
untouched_test.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',index=False,header=False)
stretched_test=stretched_datafr(untouched_test)
stretched_test.to_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',index=False,header=False)

#model:data prepare
untouched_train=pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_train.csv',sep=',')
untouched_train_Y=untouched_train["label"]
untouched_train_X=untouched_train.drop(columns=["label"])
stretched_train = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_train.csv',sep=',')
stretched_train_Y=stretched_train["label"]
stretched_train_X=stretched_train.drop(columns=["label"])

untouched_val=pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_val.csv',sep=',')
untouched_val_Y=untouched_val["label"]
untouched_val_X=untouched_val.drop(columns=["label"])
stretched_val=pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_val.csv',sep=',')
stretched_val_Y=stretched_val["label"]
stretched_val_X=stretched_val.drop(columns=["label"])

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)


#model:run
#GaussianNB
model = GaussianNB()
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,1,model)
digits_image(untouched_test_X,pred,28,1)

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,2,model)
digits_image(stretched_test_X,pred,20,2)

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)

#BernoulliNB
model = BernoulliNB()
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,3,model)
#genrate digits mean image
digits_image(untouched_test_X,pred,28,3)

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,4,model)
digits_image(stretched_test_X,pred,20,4)

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)

#RandomForest
model= RandomForestClassifier(n_estimators=10, max_depth=4)
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,5,model)
#genrate digits mean image

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,6,model)

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)

model= RandomForestClassifier(n_estimators=10, max_depth=16)
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,7,model)
#genrate digits mean image

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,8,model)

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)

model= RandomForestClassifier(n_estimators=30, max_depth=4)
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,9,model)
#genrate digits mean image

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,10,model)

untouched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/untouched_test.csv',sep=',',header=None)
stretched_test_X = pd.read_csv('/Users/panzhang/Desktop/CS498AML/HW1/stretched_test.csv',sep=',',header=None)

model= RandomForestClassifier(n_estimators=30, max_depth=16)
pred=run_model(untouched_train_X,untouched_train_Y,untouched_val_X, untouched_val_Y,untouched_test_X,11,model)
#genrate digits mean image

pred=run_model(stretched_train_X,stretched_train_Y,stretched_val_X, stretched_val_Y,stretched_test_X,12,model)
