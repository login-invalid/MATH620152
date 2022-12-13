import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import MultiTaskElasticNet

class elastic():
    def __init__(self,path):
        self.path = path
        self.data = pd.read_csv(self.path)
        self.features = set(self.data.columns)

        print(f"You are using sklearn:{sklearn.__version__}.")
        print(f"Nice to see you! This project is designed as Sklearn Class 2.")

    def elsastic_net(self,target,x,params,test_ratio = 0.,random_seed = 0):
        '''
        生成弹性网络模型 , 返回为指定模型;

        :param target(str list): 回归因变量(此时只能有一个);
        :param x(str list): 回归自变量;
        :param test_ratio(float): 0~1数值 , 测试数据比例 , 设置为0时则默认训练用全量数据;
        :param random_seed(int): 随机种子;  
        :param params(dict): 所使用的参数字典;
        :return: 拟合后的模型enet.
        '''
        # 这里采用了与之前的 sklearn 不同的架构，体现在 params 更自由定义
        if len(target) != 1:
            print("Not supported target!")
            return -1
        if set(list(x)).issubset(self.features) and set(list(target)).issubset(self.features):
            y = self.data[target]
            X = self.data[list(x)]
            if test_ratio:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)
            else:
                X_train, y_train = X,y
            print("+"*60)
            print(f"You are using n_samples: {X.shape[0]} and n_features: {X.shape[1]} to train!") 
            print(f"Hyper parameters: {list(params.keys())}")
            
            enet = ElasticNet()
            enet.set_params(**params)
            enet = enet.fit(X_train.values, y_train, sample_weight=None, check_input=True)
            # use X_train.values indstead of X_train will mute the warnings

            if test_ratio:
                print("+"*60)
                print(f"Score on test set is {enet.score(X_test.values, y_test, sample_weight=None)}")
            return enet
        else:
            print("Please check your input args!")
            return -1
    
    def multi_elsastic_net(self,target,x,params,test_ratio = 0.,random_seed = 0):
        '''
        生成多任务弹性网络模型 , 返回为指定模型;

        :param target(str list): 回归因变量;
        :param x(str list): 回归自变量;
        :param test_ratio(float): 0~1数值 , 测试数据比例 , 设置为0时则默认训练用全量数据;
        :param random_seed(int): 随机种子;  
        :param params(dict): 所使用的参数字典;
        :return: 拟合后的模型mten.
        '''
        if set(list(x)).issubset(self.features) and set(list(target)).issubset(self.features):
            y = self.data[list(target)]
            X = self.data[list(x)]
            if test_ratio:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=random_seed)
            else:
                X_train, y_train = X,y
            print("+"*60)
            print(f"You are using n_samples: {X.shape[0]} and n_features: {X.shape[1]} to train!") 
            print(f"Hyper parameters: {list(params.keys())}")
            
            mten = MultiTaskElasticNet()
            mten.set_params(**params)
            mten = mten.fit(X_train.values, y_train)

            if test_ratio:
                print("+"*60)
                print(f"Score on test set is {mten.score(X_test.values, y_test, sample_weight=None)}")
            return mten
        else:
            print("Please check your input args!")
            return -1

class Linear():
    def __init__(self,Y,k,X):
        """
        k,X 必须有一个非 None,当 X 非 None 时不会自动生成 X;
        k: 参数个数 , 包含常数项!
        """
        self.Y = Y
        self.n = len(Y)
        self.k = k
        self.X = X
        self.b = None
        self.e = None
    
    def __checker(self):
        if np.isnan(self.Y).any() | np.isnan(self.X).any():
            print("NaN exists!")
            return -1
        else:
            print("OLS Progressing...")
            return 0
        
    def __estimator(self, l2 = 0.):
        self.Y = np.matrix(self.Y)
        if self.X is None:
            self.X = self.__gen_X()
        else:
            self.X = np.matrix(self.X)
        X = self.X
        if X is None:
            return -1
        # keep Y column vector
        if self.__checker() == 1:
            return -1

        try:
            if np.shape(self.Y)[1]!= 1:
                self.Y = self.Y.T
                Y = self.Y
            tmp = np.matrix(X.T @ X)
            if l2:
                tmp = tmp + (np.identity(self.k) * l2)
            self.b = (tmp.I @ X.T)@Y # b = (X'X)^-1 X'Y
            self.e = Y - X @ self.b
            return 0
        except:
            return -1

    def __gen_X(self):
        """
        用于自动生成一组测试数据X.
        """
        if self.n < self.k:
            print("Notice: It's not supported.")
            return None
        ret = np.full((self.n,self.k),np.nan)
        for row in range(self.n):
            ret[row] = np.array([(row+1)**x for x in range(self.k)])
        return np.array(ret)

    def fit_Constrain(self,R,q):
        """
        带有限制条件的回归;
        限制条件为: Rb - q = 0;

        :param R(np.matrix): R矩阵
        :param q(np.matrix): q矩阵
        :return: 拟合参数 b.
        """
        print("You are fitting with constraints...")
        cond = self.__estimator()
        if cond == -1:
            print("Something Wrong!")
            return None
        XtX = np.matrix(self.X.T @ self.X)
        temp = np.matrix(R @ (XtX.I) @ R.T).I
        b = self.b - XtX.I @ R.T @ temp @ (R @ self.b - q)
        e = self.Y - self.X @ b
        print("F statistic = {}".format(float( ((e.T@e-self.e.T@self.e) / len(R))/((e.T@e) / (self.n-self.k)) )))
        return b
    
    def fit(self,is_detail = True, l2 = 0.):
        """
        拟合模型 , 返回拟合参数 b;

        :param is_detail(bool): True or False 是否输出详细拟合信息;
        :param l2(float): Ridge 回归的超参;
        :return: 拟合参数 b.
        """
        cond = self.__estimator(l2 = l2)
        if cond == -1:
            print("Something Wrong!")
            return None
        
        print("OLS Regression Results")
        print("="*60)
        Y_mean = np.matrix([np.mean(self.Y)])
        Y_hat = self.X @ self.b
        e = self.e
        XtX = np.matrix(self.X.T @ self.X)
        s2 = float(e.T@e/(self.n-self.k)) # s2 is MSE
        StdX = np.sqrt(s2* np.diag(XtX.I))
        
        diff = self.Y - Y_mean
        diff2 = Y_hat - Y_mean
        #print("b = \n {}".format(self.b))
        for i in range(len(self.b)):
            if i == 0:
                print("const = {} , [0.025 0.975]:[{},{}]".format(float(self.b[0]),float(self.b[0]-StdX[0]*1.96),float(self.b[0]+StdX[0]*1.96)))
                print("t test = {}".format(float(self.b[0]/StdX[0])))
            else:
                print("x_{} = {} , [0.025 0.975]:[{},{}]".format(i,float(self.b[i]),float(self.b[i]-StdX[i]*1.96),float(self.b[i]+StdX[i]*1.96)))
                print("t test = {}".format(float(self.b[i]/StdX[i])))

        print("="*60)
        print(f"Ruc = {float(Y_hat.T@Y_hat)/float(self.Y.T@self.Y)}")
        print(f"Rc = {float(diff2.T@diff2)/float(diff.T@diff)}") # SSR/SST
        
        if is_detail:
            llf = - self.n/2. * np.log(2 * np.pi) - self.n/2. * np.log(float(e.T@e) / self.n) - self.n/2. # log-liklihood function

            R = np.identity(self.k)
            R = np.delete(R,0,0)
            b_ = R@self.b
            X_ = R@np.matrix(XtX).I@R.T

            print(f"F-statistic = {float(b_.T@(X_).I@b_/(s2*(self.k-1)))}")
            print(f"Log-Likelihood = {llf}")
            print(f"AIC = {-2*llf+2*self.k}")
            print(f"BIC = {-2*llf+np.log(self.n)*self.k}")
        print("="*60)
        #print("e = \n {}".format(self.e))
        return self.b

    def predict(self,X0,Y0 = None):
        """
        使用模型进行预测;

        :param X0(list): X数据;
        :param Y0(list): Y数据 , 如果有则默认输出拟合优度;
        :return: Y0_predict(list) 预测值.
        """
        if self.b is None:
            self.fit(False)
        Y0_predict = X0 @ self.b
        s2 = float((self.e.T@ self.e)/(self.n-self.k)) # s2 is MSE
        StdY = np.identity(len(X0)) + X0 @ np.matrix(X0.T @ X0).I @ X0.T
        StdY = np.sqrt(s2 * np.diag(StdY))
        if Y0 is not None:
            Y0 = np.array(Y0).reshape(-1,1)
            print("Test validation: ===========================================")
            e0 = Y0 - Y0_predict
            print(f"RMSE = {float(np.sqrt((e0.T @ e0)/len(X0)))}")
        for i in range(len(X0)):
            print("Y_{}= {} , [0.025 0.975]:[{},{}]".format(i,float(Y0_predict[i]), \
                float(Y0_predict[i]-StdY[i]*1.96),float(Y0_predict[i]+StdY[i]*1.96)))
        return Y0_predict