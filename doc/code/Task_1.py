import pandas as pd
import numpy as np
from wordcloud import WordCloud 

import jieba
import numpy as np
from PIL import Image
import pandas as pd
from wordcloud import STOPWORDS

import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import BayesianRidge, ARDRegression
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.svm import SVC

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

import seaborn as sns

import requests
import re
import urllib3

import pandas as pd
import datetime
requests.packages.urllib3.disable_warnings()

class data_pd():
    def __init__(self,path):
        """
        :param path(str):文件读取路径;
        """
        self.path = path
        print(f"You are using pandas:{pd.__version__} and numpy:{np.__version__}")
        print(f"Nice to see you! This project is designed as Pandas Class 1.")
        self.data_all = {}
        
        self.file = None
        self.state = None
        self.schedule = None

    def read_data_all(self):
        """
        多表数据读入,多级索引结构导入;
        表的基础数据清洗 (NA处理、删除非必要数据);
        """
        if len(self.data_all) == 0:
            if (self.file is None):
                self.file = pd.read_excel(self.path,sheet_name = 'File Name',
                        header = 2, index_col = [0,1])
                        # since header is beginning from index 2
                self.file = self.file.dropna(how = 'all',axis= 0) # 数据清洗
                self.file = self.file[["Link to Fields used in the files "]].fillna(method='ffill')
                # 向前填充NA数据

            if (self.state is None):
                self.state = pd.read_excel(self.path,sheet_name = 'State District Codes',
                        header = 2, index_col = [0,1])
                        # since header is beginning from index 2
                self.state = self.state.dropna(how = 'all',axis= 0) # 数据清洗
                self.state = self.state.sort_index(level = [0,1]) # 进行排序 , 避免性能警告

            if (self.schedule is None):
                self.schedule = pd.read_excel(self.path,sheet_name = 'Schedule Codes',
                        header = 2, usecols= [1,2] ,index_col = [0])              
                self.schedule = self.schedule.dropna(how = 'all',axis= 0) # 数据清洗                
        else:
            print("Data has already been loaded!")
            return 'succeed'

        sheet_name = self.file.index.droplevel(1).unique()
        print(f"There are {len(sheet_name)} other sheets parsed in xlsx and they are:{list(sheet_name)}.")

        for name in sheet_name:
            try:
                tmp_df = pd.read_excel(self.path,sheet_name = name, header = [2,3], index_col = [0])
                # since there are multi_index in headers
                tmp_df = tmp_df.dropna(how = 'all',axis= 0) # 数据清洗

                # deal with multi_index in columns
                columns = [tuple([x,x]) for x in tmp_df.columns[:4].droplevel(1)]
                columns.extend(list(tmp_df.columns[-3:]))
                columns = pd.MultiIndex.from_tuples(columns)
                tmp_df.columns = columns

                # delete NOTES info
                tmp_df = tmp_df.drop(index = 'NOTES:')

                self.data_all[name] = tmp_df
            except Exception as e:
                print(e)
                return -1
        return 'succeed'

    def get_shape(self):
        """
        输出所有表的结构信息;
        """
        print('='*60)
        for sheet in self.data_all.keys():
            print(f"Sheet {sheet}'s shape is {self.data_all[sheet].shape}")
        return

    def show_data(self,df,n = 5):
        """
        :param df(DataFrame): 输入的数据框;
        :param n(int): 展示数据的行数;
        """
        print('='*60)
        print(f"Columns: {list(df.columns)}")
        print(df.head(n))
        return
    
    def yellow_na(self,name):
        """
        处理黄色无用数据;

        :param name(str): 待处理数据框的名字;
        :return: 是否成功的信息.
        """
        if name not in ['COMB', 'MORT', 'WOMAN', 'WPS']:
            print("Undefined dataframe!")
            return -1
        else:
            if name not in self.data_all.keys():
                print("Undefined dataframe!")
                return -1
            df = self.data_all[name]
            yellow_mask = df.iloc[:,1:].isna().all(1)
            print(f"There are {sum(yellow_mask)} data can be ignored since they are marked yellow.")
            self.data_all[name] = self.data_all[name].iloc[list(~yellow_mask)] # filter out these data
        return f'Filtered Out Yellow Marked Data for {name} sheet.'
    
    def black_na_ret(self,name):
        """
        处理有黑色标注的数据;

        :param name(str): 待处理数据框的名字;
        :return: ret(DataFrame): 去除有黑色标注的数据后的数据框.
        """
        if name not in ['COMB', 'MORT', 'WOMAN', 'WPS']:
            print("Undefined dataframe!")
            return -1
        else:
            df = self.data_all[name]

            # 黑色数据逻辑
            black_mask_1 = df.iloc[:,1:-3].isna().any(1)
            black_mask_2 = df.iloc[:,-3:].isna().any(1)
            black_mask = (~black_mask_1) & (black_mask_2)

            print(f"There are {sum(black_mask)} data can be ignored since they are marked black.")
            ret = self.data_all[name].iloc[list(black_mask)] # filter out these data
        print(f'Filtered Out black Marked Data for {name} sheet.')
        return ret
    
    def one_time_yellow(self):
        """
        一次性处理所有数据框中的黄色无用数据;
        """
        print('='*60)
        for name in ['COMB', 'MORT', 'WOMAN', 'WPS']:
            state = self.yellow_na(name)
            if state == -1:
                print(f"Something went wrong for {name} sheet!")
        return 'filtration succeed'
    
    def search_district(self,state = 18,district = 1):
        """
        根据给定的区号查询区名;

        :param state(int): state 编号;
        :param district(int): district 编号;
        :return: name(str): 区名.
        """
        name = None
        print('='*60)
        try:
            name = self.state.loc[pd.IndexSlice[(state, district)]].iloc[0,0].rstrip()
            print(f"State Code {state} District Code {district} is {name}.")
        except:
            print("ID not found!")
        return name

    def merge_data(self,name):
        """
        根据给定的表名合并 Schedule 中的 description;
        数据框整合 (merge连接);

        :param name(str): 待处理数据框的名字;
        :return: df(DataFrame): 合并后的数据框.
        """
        print('='*60)
        if name not in ['COMB', 'MORT', 'WOMAN', 'WPS']:
            print("Undefined dataframe!")
            return -1
        else:
            if name not in self.data_all.keys():
                print("Undefined dataframe!")
                return -1
            df = self.data_all[name] # 实际是创建副本 , 不会影响原始数据
            df = df.droplevel(0,axis=1)
            df = df.merge(data.schedule,on = 'Schedule Code',how = 'left')
            print("Merge Succeed!")
        return df
    
    def output(self,df, file_name, is_csv = True):
        """ 
        :param df(DataFrame): 待保存的数据框;
        :param file_name(str): 保存文件名,不用带后缀;
        :param is_csv(Bool): 是否保存为csv格式 ,否则存为xlsx格式;
        """
        if is_csv:
            file_name += '.csv'
            df.to_csv(file_name,header = True, index = True)
        else:
            file_name += '.xlsx'
            df.to_excel(file_name,header = True, index = True)      
        return

class data_cloud():
    def __init__(self,path):
        self.path = path
        print(f"You are using jieba:{jieba.__version__}.")
        print(f"Nice to see you! This project is designed as WorldCloud Class.")

        self.english = None
        self.chinese = None
        self.gen()

    def gen(self): 
        """ 
        txt 文件读入 (中英文自动识别);
        """
        self.english = []
        self.chinese = []
        n,i = 0,0
        try:
            fp = open(self.path,mode = "r", encoding = "utf-8")
            for x in fp:
                n += 1
                if x.strip():
                    self.english.append(x.split('\n')[0])
        except Exception as e:
            fp.close()
            fp = open(self.path,mode = "r", encoding="gbk")
            for x in fp:
                if i<=n+10: # 有10行未知错误
                    i += 1
                    continue
                if x.strip():
                    self.chinese.append(x.split('\n')[0])
        else:
            fp.close()
            print("Something went wrong!")
            return -1
        finally:
            fp.close()
            print("File Closed!")

        self.english = ' '.join(self.english)
        self.chinese = ' '.join(self.chinese)
        return 'succeed'
    
    def __type_txt(self,type):
        """ 
        根据 type 返回对应 txt 存储;
        """
        if type == 'english':
            return self.english
        elif type == 'chinese':
            return self.chinese
        else:
            print("Wrong Type!")
            return -1

    def show_text(self,type = 'english',n=300):
        """
        展示指定文档;

        :param type(str): 'english' or 'chinese';
        :param n(int): 展示文章前 n 个词;
        """
        txt = self.__type_txt(type)
        print('+'*60)        
        print(txt[:n])
        print('+'*60)
        print(f"The txt's length is {len(txt)}.")
        return
    
    def gen_wordcloud(self,type = 'english',n=300, file_name = "wcd1.png",max_words = 250):
        """ 
        依据指定文档前 n 个词语生成词云图片;
        
        :param type(str): 'english' or 'chinese';
        :param n(int): 文章前 n 个词;
        :param file_name: 存储图片文件名;
        :param max_words: 词云提取最大文字数量;
        :return: 是否成功的信息.
        """
        txt = self.__type_txt(type)
        if type == 'english':
            wcd = WordCloud(max_words = max_words,width = 700,height = 350)
            wcd.generate(txt[:n])
            wcd.to_file(file_name)
        elif type == 'chinese':
            wcd = WordCloud(font_path = "./support/data_cloud/msyh.ttc",background_color = "White"
                            ,max_words = max_words, width = 700,height = 350)
            # Notice: the font file is must for chinese wordcloud!
            ss = " ".join(jieba.lcut(txt[:n]))
            wcd.generate(ss)
            wcd.to_file(file_name)
        return 'succeed'

    def mask(self,type = 'english', pic = "./support/data_cloud/pic2.png",n=300,file_name = "wcd3.png",max_words = 250):
        """
        mask 词云图像: 将从 pic 路径读取 mask 图片;

        :param type(str): 'english' or 'chinese';
        :param pic(str): mask图片路径;
        :param n(int): 采用文档前 n 个词;
        :param file_name(str): 保存图片路径;
        :param max_words(int): 词云图像中最大词个数;
        :return: 是否成功的信息.
        """
        txt = self.__type_txt(type)
        mask = np.array(Image.open(pic))

        #############################################
        # 以下代码将造成大量GPU占用，且del变量未能释放占用 #
        #############################################
        if type == 'english':
            wcd = WordCloud(mask=mask, max_words = max_words,
                            contour_width = 2, contour_color = "Pink")
            wcd.generate(txt[:n])
            wcd.to_file(file_name)
        elif type == 'chinese':
            wcd = WordCloud(font_path="./support/data_cloud/msyh.ttc", mask=mask, background_color="white",
                            contour_width = 3, contour_color = "Pink", max_words = max_words) 
            ss = " ".join(jieba.lcut(txt[:n]))
            #print(ss)
            wcd.generate(ss)
            wcd.to_file(file_name)
        return 'succeed'
    
    def stop_words_mask(self,n=300,file_name = "wcd3.png",max_words = 250):
        """
        mask 词云图像: 带有停词库过滤;

        :param n(int): 采用文档前 n 个词;
        :param file_name(str): 保存图片路径;
        :param max_words(int): 词云图像中最大词个数;
        :return: 是否成功的信息.
        """
        stopwords = set()
        # 更新中文停用词库
        content = [line.strip() for line in open('./support/data_cloud/cn_stopwords.txt','r',encoding='UTF-8').readlines()]
        stopwords.update(content)

        mask = np.array(Image.open("./support/data_cloud/pic2.png"))
        wcd = WordCloud(font_path="./support/data_cloud/msyh.ttc",mask=mask,background_color="white",
                        scale = 1 , max_font_size = 150 , min_font_size = 10,
                        stopwords = stopwords, colormap="spring", max_words = max_words)
        ss = " ".join(jieba.lcut(self.chinese[:n]))
        wcd.generate(ss)
        wcd.to_file(file_name)
        return 'succeed'

def bayes_plot(model,X,y,file_name="output_fig.png"):
    """ 
    贝叶斯回归的图示;

    :param model: 已完成训练的模型;
    :param X(DataFrame): 预测数据;
    :param y(DataFrame): 预测标签;
    :param file_name(str): 图片保存路径;
    :return: 是否成功的信息.
    """
    y_brr, y_brr_std = model.predict(X, return_std=True)
    full_data = pd.DataFrame({"input_feature": X[:,0], "target": y})
    
    ax = sns.scatterplot(
        data=full_data, x="input_feature", y="target", color="black", alpha=0.75
    )
    ax.plot(X[:,0], y, color="black", label="Ground Truth")
    ax.plot(X[:,0], y_brr, color="red", label="BayesianRidge with polynomial features")
    ax.fill_between(
        X[:,0].ravel(),
        y_brr - y_brr_std,
        y_brr + y_brr_std,
        color="red",
        alpha=0.3,
    )
    ax.legend()
    ax = ax.set_title("Polynomial fit of a non-linear feature")
    ax.get_figure().savefig(file_name)
    return 'succeed'

class data_learn():
    def __init__(self,path):
        self.path = path
        print(f"You are using sklearn:{sklearn.__version__}.")
        print(f"Nice to see you! This project is designed as Sklearn Class.")

        self.data = pd.read_csv(self.path,index_col=[0])
        self.features = set(self.data.columns)
        self.tempX = None # 使用 gen_data 方法后会同步最后一次的X_train , y_train
        self.tempy = None
    
    def gen_data(self,formula, ratio = 0.1, is_scale = False):
        """ 
        用formual (str格式 )形式获取测试训练数据;

        :param formula(str): 回归公式;
        :param ratio(float): 测试数据比例 , 设置为0时则默认训练用全量数据;
        :param is_scale(bool): True or Flase 是否对数据进行归一化处理;
        :return: 根据需求返回生成数据集(DataFrame).
        """
        type = len(formula.split('~'))
        if type == 1: # 没有 y 值数据 , 当前为无监督学习
            print("Notice: This is unsupervised learning since y is omitted!")
            X_name = formula.split('~')[0].split('+')
            if set(X_name).issubset(self.features):
                if ratio:
                    r = int(self.data.shape[0]*ratio)
                    index = np.random.permutation(self.data[X_name].shape[0]) # 利用index随机打乱数据
                    index1 = index[:r]
                    index2 = index[r:]

                    X_train = np.array(self.data[X_name].iloc[index1])
                    X_test = np.array(self.data[X_name].iloc[index2])

                    if is_scale:
                        X_train = StandardScaler().fit_transform(X_train)
                        X_test = StandardScaler().fit_transform(X_test)
                    return X_train,X_test
                else:
                    X_train = np.array(self.data[X_name])
                    if is_scale:
                        X_train = StandardScaler().fit_transform(X_train)
                    return X_train


        y_name = formula.split('~')[0].split('+')
        if len(y_name)>=2:
            print("y should be 1-d array!")
            return -1
        X_name = formula.split('~')[1].split('+')
        if set(X_name).issubset(self.features) and set(y_name).issubset(self.features):
            if ratio:
                r = int(self.data.shape[0]*ratio)
                index = np.random.permutation(self.data[X_name].shape[0]) # 利用index随机打乱数据
                index1 = index[:r]
                index2 = index[r:]

                X_train = np.array(self.data[X_name].iloc[index1])
                y_train = np.array(self.data[y_name].iloc[index1]).ravel()
                X_test = np.array(self.data[X_name].iloc[index2])
                y_test = np.array(self.data[y_name].iloc[index2]).ravel()
                if is_scale:
                    X_train = StandardScaler().fit_transform(X_train)
                    X_test = StandardScaler().fit_transform(X_test)
                return X_train,y_train,X_test,y_test
            else: # ratio == 0
                X_train = np.array(self.data[X_name])
                y_train = np.array(self.data[y_name]).ravel()
                if is_scale:
                    X_train = StandardScaler().fit_transform(X_train)
                return X_train,y_train


    def bayes_ridge_re(self, formula,degree=10,bias = False,test_ratio=0.,is_ARD = False,is_scale = False):
        """ 
        贝叶斯回归模型拟合 , 返回为指定模型;

        :param formula(str): 回归公式;
        :param degree(int): 模型参数 , 多项式拟合阶数;
        :param bias(bool): True or False 是否采用偏置项;
        :param test_ratio(float): 0~1数值 , 测试数据比例 , 设置为0时则默认训练用全量数据;
        :param is_ARD(bool): True or False 采用ARD回归或者贝叶斯回归;
        :param is_scale(bool): True or Flase 是否对数据进行归一化处理;
        :return: 拟合后的模型.
        """
        if test_ratio:
            X,y,_, _ = self.gen_data(formula,test_ratio, is_scale= is_scale)
            self.tempX,self.tempy = X,y
        else:
            X,y = self.gen_data(formula,test_ratio, is_scale= is_scale)
            self.tempX,self.tempy = X,y
        if is_ARD:
            ard_poly = make_pipeline(
                PolynomialFeatures(degree = degree, include_bias = bias),
                StandardScaler(),
                ARDRegression(),
            ).fit(X, y)
            return ard_poly
        else:
            brr_poly = make_pipeline(
                PolynomialFeatures(degree = degree, include_bias = bias),
                StandardScaler(),
                BayesianRidge(),
            ).fit(X, y)
            return brr_poly

    def cluster_model(self,formula,type = 'kmean',random_seed = 0, is_scale = False,**kwarg):
        """ 
        聚类方法集成 , 返回为指定模型;

        :param formula(str): 回归公式;
        :param type(str): 聚类模型类型 , 默认为 kmean;
        :param random_seed(int): 随机种子;
        :param kwarg: 传入模型必须参数;
        :param is_scale(bool): True or Flase 是否对数据进行归一化处理；
        :return: 拟合后的模型.
        """
        # 目前 kwarg 只能集成了不同模型的一个主要关键字，因为只研究了这2个参数
        X = self.gen_data(formula, ratio = 0., is_scale= is_scale)
        self.tempX = X

        if type == 'kmean':
            kmeans = KMeans(n_clusters = kwarg['n_cluster'],random_state=random_seed)
            kmeans = kmeans.fit(X)
            print(f"Kmeans Score is {kmeans.score(X)}")
            return kmeans
        elif type == 'DBSCAN':
            DBCSN = DBSCAN(eps = kwarg['eps'])
            DBCSN = DBCSN.fit(X)
            return DBCSN           

        else:
            print("Not supported!")
            return -1
    
    def svm_svc(self,formula, kernal = 'linear',test_ratio=0., is_scale = False, **kwarg):
        """
        支持向量机方法 , 返回为指定模型;

        :param formula(str): 回归公式;
        :param kernal(str): SVC支持向量机的核函数;
        :param test_ratio(float): 0~1数值 , 测试数据比例 , 设置为0时则默认训练用全量数据;
        :param is_scale(bool): True or Flase 是否对数据进行归一化处理；
        :return: 拟合后的模型svc.
        """
        if kernal not in ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']:
            print("Not supported kernal")
            return -1
        
        if test_ratio:
            X,y,X2, y2 = self.gen_data(formula,test_ratio, is_scale= is_scale)
            self.tempX,self.tempy = X,y
        else:
            X,y = self.gen_data(formula,test_ratio, is_scale= is_scale)
            self.tempX,self.tempy = X,y
        
        if 'gamma' in kwarg.keys():
            svc = SVC(kernel = kernal, gamma = kwarg['gamma']).fit(X, y)
        else:
            svc = SVC(kernel = kernal).fit(X, y)

        if test_ratio:
            print(f"Self predict score is {svc.score(X2, y2)}")
        return svc
    
    def search_hyper_para(self,model, param_grid, n_split = 5,random_seed = 0):
        """ 
        对给定的模型进行超参搜索 , 采用 K 折检验; 不支持无监督学习！

        :param model: 待搜索的模型；
        :param param_grid(list): 待搜索的参数列表;
        :param n_splits(int): K 折检验的 K 值; 
        :param random_seed(int): 随机种子;
        :return: grid.best_params_最优超参列表.
        """
        print("Notice: Please establish the model first!")
        if self.tempX is None:
            return -1
        
        cv = StratifiedShuffleSplit(n_splits = n_split, test_size = 0.3, random_state = random_seed)
        
        grid = GridSearchCV(model, param_grid = param_grid, cv = cv)
        grid.fit(self.tempX, self.tempy)
        
        print(f"The best parametes are {grid.best_params_} with a score of {grid.best_score_}")
        return grid.best_params_

class WebSchedules():
    def __init__(self, name, web):
        """
        :param name(str): 对象存储名字;
        :param web(str): 浏览器种类;
        :params chedules: 爬取的日志;
        :param schedules_stored: 存储的日志;
        """
        self.name = name
        self.web = web
        self.schedules = {'date': [], 'version': []}
        self.schedules_stored =''
        self.processed_schedules = pd.DataFrame(columns= ('date','version'))

        print(f"You are using requests:{requests.__version__}.")
        print(f"Hi~Yi. This class is designed for getting chrome-version schedule!")

    def getschedules(self, item):
        """
        :param item(json): json 文件对象, 从中解析日期、版本;
        """
        if self.web == 'chrome':
            self.schedules['date'].append(item['start']['date'])
            self.schedules['version'].append(item['summary'])
        elif self.web == 'firefox':
            self.schedules['date'].append(item['start']['dateTime'].split('T')[0])
            self.schedules['version'].append(item['summary'])
        return 'succeed'

    def processtschedules(self, isunique= True, isprint = True):
        """
        :param isunique(bool): 如果为 True 则版本重复发布 , 只保留一条信息;
        :param isprint(bool): 是否输出日志信息;
        """
        self.processed_schedules = pd.DataFrame(self.schedules)
        self.processed_schedules['date'] = pd.to_datetime(self.schedules['date'])
        self.processed_schedules = self.processed_schedules.sort_values('date')
        self.processed_schedules = self.processed_schedules.set_index('date')
        if isunique:
            self.processed_schedules = self.processed_schedules.drop_duplicates(subset=['version'])
        if isprint:
            print("You are processing: {}".format(self.name))
            print(self.processed_schedules.to_string())
        return 'succeed'

    def updateschedules_stored(self):
    # Update schedule when different
        """
        更新 schedule;
        """
        schedule_print = '\n' + self.web.upper() + ' SCHEDULE:\n' + self.processed_schedules.to_string()
        now = datetime.datetime.now()
        if self.schedules_stored != schedule_print:
            self.schedules_stored = schedule_print
            print('\n', now, '\nUPDATE:', self.schedules_stored)
        else:
            print('SAME '+ self.web.upper() +' SCHEDULE.', now)
        return 'succeed'

    
    def updateschdeules(self,checkdate = datetime.date(2022, 1, 1)):
        """
        :param check_date(datetime.date): 查找日期 , 如果在那天有新版本发布则会显示;
        """
        for date in  pd.to_datetime(self.processed_schedules.index.values.tolist()):
            if 0 == (date - checkdate).days:         
                web_sr = self.processed_schedules[self.processed_schedules.index==date]['version']
                # print('{} RELEASED TODAY.'.format(web_sr.values[0]))
                notified_schedule = self.web.upper()+ ' ' + web_sr.values[0] + ' RELEASED TODAY! '
                print(notified_schedule)
                break
        return 'succeed'