import numpy
import scipy
import pandas
import os
import time
import math
import networkx as nx
from matplotlib import pyplot as plt
class JudgeTree:
    data = []
    Ent_name = 0
    Attribute = ["man", "work", "70", "bald", "80", "marr", "selc", "bas", "land", "actor"]
    lib = ['yaoming 1 1 0 0 1 0 0 1 1 0','liuxinag 1 1 0 0 1 1 0 0 1 0','kebi 1 1 1 1 0 0 0 1 0 0','Cluo 1 1 0 0 1 0 0 0 0 0','liudehua 1 0 0 0 0 0 0 0 0 1','maobuyi 1 0 0 0 0 0 1 0 1 0','zhoujielun 1 0 1 0 0 0 0 0 0 1','huangbo 1 0 1 0 0 0 0 0 1 1','xuzheng 1 0 1 1 0 0 0 0 1 1','zhangyining 0 1 0 0 1 0 0 0 1 0','langping 0 1 0 0 0 1 0 0 1 0','zhuting 0 1 0 0 0 0 0 0 1 0','yangchaoyue 0 0 0 0 0 0 1 0 1 1','yangmi 0 0 0 0 1 1 0 0 1 1','degnziqi 0 0 0 0 0 0 0 0 0 0','xujiaying 0 0 0 0 1 0 1 0 0 0','zhaoliying 0 0 0 0 1 0 0 0 1 1']
    G = nx.DiGraph()
    pos_data = {}
    lable_count = 0

    def Input(self):
        #with open("C:\E盘\大三作业\大三下作业\机器学习\第一次大作业\信息1.txt") as file:
        for info in self.lib:
            temp = info.split()
            a = {}
            a["name"] = temp[0]
            for i, count in zip(self.Attribute,range(1,len(temp))):
                a[i] = temp[count]
            self.data.append(a)
        self.pos_data['root'] = (20, 0)

    def Possiblity(self, attr, data):  # 返回一个该属性下所有属性种类名称的列表:choice = 选择的属性， data = 集合
        temp = []
        for i in data:
            if i[attr] not in temp:
                temp.append(i[attr])
        return temp

    def Ent(self, data):
        length = len(data)
        if length <= 1:
            return 0
        temp1 = 1/length * math.log2(1/length)
        temp2 = (length - 1) * math.log2((length - 1) / length)
        temp = (temp1 + temp2) * -1 * length
        return temp

    def GianCount(self, data, attr):  # 集合D的信息增量计算函数，data：集合，attr：属性.返回值为该属性的Gain
        length = len(data)
        # 节点的信息熵
        Ent_data = self.Ent(data)  # 该节点未划分的时候，ent的值

        # 对该节点遍历属性，寻找最优属性
        attrson = self.Possiblity(attr, data)  # 获得在该属性下所有属性种类，以及每个种类下的数据个数

        #遍历所有的亚种属性，对传入的集合D进行划分
        for i in attrson:
            tempdata = []
            for info in data:
                if info[attr] == i:  #传入的数据集中，当前数据的该属性恰为这个亚种
                    tempdata.append(info)
            # D中所有为该亚种的数据已统计完毕，开始计算Ent,然后乘以对应的加权比例
            Entson = self.Ent(tempdata)
            Entson = Entson * (len(tempdata)/length)
            Ent_data -= Entson

        return Ent_data

    def Select(self, data, last, x, y, dis):
        # 如果当前传入的数据只有1个或者已经没有符合的数据，则返回
        if len(data) <= 1 or (len(self.Attribute) <= 0):
            print("END ", data, "\n")
            return
        MaxGain = 0
        BestAttr = "name"
        for i in self.Attribute:
            temp = self.GianCount(data, i)
            print("当前检测的属性为： ", i, " Ent值为： ", temp)
            if temp > MaxGain:
                MaxGain = temp
                BestAttr = i
        print("最佳属性为：", BestAttr, " Gain为： ", MaxGain, "此时该分支下还有", len(data), "个数据")

        # 找到最佳属性之后，分类,递归继续划分
        if BestAttr != "name":
            self.Attribute.remove(BestAttr)
        tempdata = self.Possiblity(BestAttr, data)
        copy = self.lable_count
        for i, count in zip(tempdata, range(1, 3)):
            #self.G.add_node(BestAttr + "." + i)
            self.pos_data[str(self.lable_count) + BestAttr + "." + i] = (x + dis*(-1)**count, y - 1)
            #  G.add_node(BestAttr + "." + i, pos=(x + dis*(-1)**count, y - 1))
            self.G.add_edge(last, str(self.lable_count) + BestAttr + "." + i)
            self.lable_count += 1
        #nx.draw_networkx(self.G)
        #plt.show()
        #time.sleep(1)
        #plt.close()
        for i, count in zip(tempdata, range(1, 3)):
            tempList = []
            for info in data:
                if info[BestAttr] == i:
                    tempList.append(info)
            self.Select(tempList, str(copy) + BestAttr + "." + i, x + dis*(-1)**count, y-1, dis/2)
            copy += 1
        self.Attribute.append(BestAttr)
        return

    def run(self):
        self.Input()
        self.Select(self.data, "root", 20, -1, 20)
        fig, ax = plt.subplots(figsize=(20, 15))
        nx.draw_networkx(self.G, self.pos_data, with_labels=True, node_color='g')
        plt.show()


test = JudgeTree()
test.run()




