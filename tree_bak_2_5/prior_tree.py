#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 17:28:15 2018

@author: zhengzhe
"""

import pandas as pd
import numpy as np
from xml.etree import ElementTree as ET
from sklearn.metrics import roc_auc_score
import networkx as nx

class prior_tree:
    #def __init__(self):
    def expected_gain(self,n1=0,N1=1,n2=2,N2=3,prior_num=1, prior_den=2, priord_num=1):
        n1 = np.float(n1)
        N1 = np.float(N1)
        n2 = np.float(n2)
        N2 = np.float(N2)
        n = n1+n2
        N = N1+N2
        P1 = (n1+prior_num)/(N1+prior_den)
        P2 = (n2+prior_num)/(N2+prior_den)
        Pd = (N1+priord_num)/(N+2*priord_num)
        P = (n+prior_num)/(N+prior_den)
        Loss_p = -P*np.log(P)-(1-P)*np.log(1-P)
        Loss_p1 = -P1*np.log(P1)-(1-P1)*np.log(1-P1)
        Loss_p2 = -P2*np.log(P2)-(1-P2)*np.log(1-P2)
        Loss_l = Pd*(Loss_p-Loss_p1)
        Loss_r = (1-Pd)*(Loss_p-Loss_p2)
        #Loss_p_s=Pd*Loss_p1+(1-Pd)*Loss_p2
        return (Loss_l,Loss_r)

    def info_gain_cal(self,col,labels,min_leaf=1,method='prior',task='classification',min_split = 4, 
                      prior_num =1, prior_den = 2, priord_num = 1):
        df=pd.DataFrame(col)
        df.columns=['feature']
        df['label']=labels
        df=df.sort_values(by='feature',ascending=True)
        if len(set(labels))<2:
            return pd.Series([-200,-100,-100,0,0,0,0,0])
        expected_info_gain_list=[]
        values=sorted(df.feature.unique())
        if len(values)<min_split:
            return pd.Series([-200,-100,-100,0,0,0,0,0])
        for i in range(len(values)-1):
            split_point=np.mean([values[i],values[i+1]])
            l_df=df[df['feature']<split_point]
            r_df=df[df['feature']>=split_point]
            n1=l_df.sum()['label']
            N1=len(l_df)
            n2=r_df.sum()['label']
            N2=len(r_df)
            if task=='classification':
                if method=='prior':
                    l_score=(n1+prior_num)/(N1+prior_den)
                    r_score=(n2+prior_num)/(N2+prior_den)
                    info_gain_l,info_gain_r=self.expected_gain(n1,N1,n2,N2,prior_num,prior_den,priord_num)
                    info_gain=info_gain_l+info_gain_r
                elif method=='prior_ratio':
                    l_score=(n1+prior_num)/(N1+prior_den)
                    r_score=(n2+prior_num)/(N2+prior_den)
                    info_gain_l,info_gain_r=self.expected_gain(n1,N1,n2,N2,prior_num,prior_den,priord_num)
                    info_all=(n1+n2+prior_num)/(N1+N2+prior_den)
                    info_gain_l=info_gain_l/info_all
                    info_gain_r=info_gain_r/info_all
                    info_gain=info_gain_l+info_gain_r
                elif method=='gini':
                    l_score=n1/N1
                    r_score=n2/N2
                    info_gain_l=2*N1/(N1+N2)*((n1+n2)/(N1+N2)*(1-(n1+n2)/(N1+N2))-l_score*(1-l_score))
                    info_gain_r=2*N2/(N1+N2)*((n1+n2)/(N1+N2)*(1-(n1+n2)/(N1+N2))-r_score*(1-r_score))
                    #info_gain=-2*N1/(N1+N2)*l_score*(1-l_score)-2*N2/(N1+N2)*r_score*(1-r_score)+2*(n1+n2)/(N1+N2)*(1-(n1+n2)/(N1+N2))
                    info_gain=info_gain_l+info_gain_r
                elif method=='prior_gini':
                    l_score=(n1+1)/(N1+2)
                    r_score=(n2+1)/(N2+2)
                    info_gain_l,info_gain_r=self.expected_gain(n1,N1,n2,N2)
                    info_gain=info_gain_l+info_gain_r                   
                elif method=='entropy':
                    l_score=n1/N1
                    r_score=n2/N2
                    if (n1+n2==0) or (n1+n2==N1+N2):
                        lg_all=0
                    else:
                        lg_all=-(n1+n2)/(N1+N2)*np.log((n1+n2)/(N1+N2))-(N1+N2-n1-n2)/(N1+N2)*np.log(1-(n1+n2)/(N1+N2))
                    if (n1==0) or (n1==N1):
                        lg_l=0
                    else:
                        #lg_l=-n1/N1*np.log(n1/N1)-(N1-n1)/N1*np.log(1-n1/N1)
                        lg_l=N1/(N1+N2)*(lg_all+n1/N1*np.log(n1/N1)+(N1-n1)/N1*np.log(1-n1/N1))
                    if (n2==0) or (n2==N2):
                        lg_r=0
                    else:
                        #lg_r=-n2/N2*np.log(n2/N2)-(N2-n2)/N2*np.log(1-n2/N2)
                        lg_r=N2/(N1+N2)*(lg_all+n2/N2*np.log(n2/N2)+(N2-n2)/N2*np.log(1-n2/N2))
                    #info_gain=-N1/(N1+N2)*lg_l-N2/(N1+N2)*lg_r+lg_all
                    info_gain=lg_l+lg_r
                    info_gain_l=lg_l
                    info_gain_r=lg_r
                elif method=='entropy_gain_ratio':
                    l_score=n1/N1
                    r_score=n2/N2
                    if (n1+n2==0) or (n1+n2==N1+N2):
                        lg_all=0
                    else:
                        lg_all=-(n1+n2)/(N1+N2)*np.log((n1+n2)/(N1+N2))-(N1+N2-n1-n2)/(N1+N2)*np.log(1-(n1+n2)/(N1+N2))
                    if (n1==0) or (n1==N1):
                        lg_l=0
                    else:
                        lg_l=-n1/N1*np.log(n1/N1)-(N1-n1)/N1*np.log(1-n1/N1)
                        lg_l=N1/(N1+N2)*(lg_all-lg_l)
                    if (n2==0) or (n2==N2):
                        lg_r=0
                    else:
                        lg_r=-n2/N2*np.log(n2/N2)-(N2-n2)/N2*np.log(1-n2/N2)
                        lg_r=N2/(N1+N2)*(lg_all-lg_r)
                    #info_gain=-N1/(N1+N2)*lg_l-N2/(N1+N2)*lg_r+lg_all
                    info_gain=lg_l+lg_r
                    if (n1+n2==0) or (n1+n2==N1+N2):
                        info_gain=-1
                    else:
                        info_gain=info_gain/lg_all
                        info_gain_l=lg_l/lg_all
                        info_gain_r=lg_r/lg_all
                else:
                    return 'error'
            elif task=='regression':
                if method=='prior':
                    mu1=n1/N1
                    mu2=n2/N2
                    if (N1>max(3,min_leaf)) and (N2>max(3,min_leaf)):
                        l_score=mu1
                        r_score=mu2
                        var_all=np.var(df['label'].values)*(N1+N2)/(N1+N2-3)
                        var_l=(N1+1)/(N1+N2+2)*(var_all-N1/(N1-3)*np.var(l_df['label'].values))
                        var_r=(N2+1)/(N1+N2+2)*(var_all-N2/(N2-3)*np.var(r_df['label'].values))
                        #var_gain=var_all-N1/(N1+N2)*var_l-N2/(N1+N2)*var_r
                        var_gain=var_l+var_r
                    else:
                        l_score=mu1
                        r_score=mu2
                        var_l=-100
                        var_r=-100
                        var_gain=-200
                    info_gain_l=var_l
                    info_gain_r=var_r
                    info_gain=var_gain
                elif method=='normal':
                    l_score=n1/N1
                    r_score=n2/N2
                    info_gain_l=N1/(N1+N2)*(np.var(df['label'].values)-np.var(l_df['label'].values))
                    info_gain_r=N2/(N1+N2)*(np.var(df['label'].values)-np.var(r_df['label'].values))
                    info_gain=np.var(df['label'].values)-N1/(N1+N2)*np.var(l_df['label'].values)-N2/(N1+N2)*np.var(r_df['label'].values)
            if (N1<min_leaf) or (N2<min_leaf):
                info_gain_l=-100
                info_gain_r=-100
                info_gain=info_gain_l+info_gain_r
            expected_info_gain_list.append((info_gain,info_gain_l,info_gain_r,split_point,l_score,r_score,N1,N2))
        return pd.Series(max(expected_info_gain_list))


    def fit(self,features,labels,max_depth=1000,min_leaf=1,min_info_gain=0,min_split=2,method='prior',
            task='classification',merge='False',greedy=False,max_greedy_times=4,min_greedy_impurity_increase=0,
            prior_num = 1, prior_den = 2, priord_num = 1,
            evaluation=False,eval_set=[],eval_func=roc_auc_score,pt_rd=5):
        self.max_depth=max_depth
        self.min_leaf=min_leaf
        self.min_info_gain=min_info_gain
        self.min_split=min_split
        self.method=method
        self.task=task
        self.min_greedy_impurity_increase=min_greedy_impurity_increase
        self.prior_num = prior_num
        self.prior_den = prior_den
        self.priord_num = priord_num
        self.pt_rd=pt_rd
        self.root=ET.Element('DecisionTree')
        self.features=features
        self.labels=labels
        self.node=self.root
        self.node=ET.SubElement(self.node,'no_split',{'value':'no_split',"flag":"no_split","level":'0',
                                                      "p_samples":str(sum(self.labels)),
                                                      'samples':str(len(self.labels)),"expected_info_gain":'0'})
        self.total_samples=len(self.labels)
        selected_samples_list=[{'selected_indexes':list(self.features.index),'node':self.node,'level':0}]
        finished_samples_list=[]
        if max_depth<0:
            max_depth=1000
        for level in range(max_depth):
            selected_samples_list_level=[s for s in selected_samples_list if s.get('level')==level]
            if selected_samples_list_level!=[]:
                for selected_samples_info in selected_samples_list_level:
                    selected_samples=selected_samples_info.get('selected_indexes')
                    self.node=selected_samples_info.get('node')
                    selected_labels=self.labels[selected_samples]
                    selected_features=self.features.loc[selected_samples]
                    expected_info_gains=selected_features.apply(lambda col:self.info_gain_cal(col,selected_labels,
                                                                                              self.min_leaf,
                                                                                              self.method,
                                                                                              self.task,
                                                                                              self.min_split,
                                                                                              self.prior_num,
                                                                                              self.prior_den,
                                                                                              self.priord_num), axis=0)
                    split_feature=expected_info_gains.idxmax(axis=1)[0]
                    expected_info_gain,expected_info_gain_l,expected_info_gain_r,split_point,l_score,r_score,N1,N2=tuple(expected_info_gains[split_feature].values)
                    expected_info_gain = expected_info_gain*(N1+N2)/self.total_samples
                    expected_info_gain_l = expected_info_gain_l*(N1+N2)/self.total_samples
                    expected_info_gain_r = expected_info_gain_r*(N1+N2)/self.total_samples
                    l_son_index=selected_features[selected_features[split_feature]<split_point].index.tolist()
                    r_son_index=selected_features[selected_features[split_feature]>=split_point].index.tolist()
                    if expected_info_gain>self.min_info_gain:
                        new_level=level+1
                        l_son=ET.SubElement(self.node,split_feature,{'value':str(split_point),"flag":"l","level":str(level),
                                                                     "p":str(l_score),
                                                                     "expected_info_gain":str(expected_info_gain_l)})
                        r_son=ET.SubElement(self.node,split_feature,{'value':str(split_point),"flag":"r","level":str(level),
                                                                     "p":str(r_score),
                                                                     "expected_info_gain":str(expected_info_gain_r)})
                        selected_samples_list.append({'selected_indexes':l_son_index,'node':l_son,'level':new_level})
                        selected_samples_list.append({'selected_indexes':r_son_index,'node':r_son,'level':new_level})
                        #print ('expected_info_gain: {0}, level: {1}, l_len: {2}, r_len: {3}'.format(expected_info_gain,level, len(l_son_index), len(r_son_index)))
                    else:
                        if greedy==False:
                            finished_samples_list.append(selected_samples_info)
                            #print(finished_samples_list)
                        elif expected_info_gain!=-200 and l_son_index!=[] and r_son_index!=[]:
                            greedy_list=self.greedy_search(l_son_index,r_son_index,split_feature,split_point,self.node,
                                                           level,max_greedy_times,N1,N2,expected_info_gain_l,expected_info_gain_r,
                                                           l_score,r_score)
                            if greedy_list!='no_positive_found':
                                selected_samples_list+=greedy_list
                if evaluation==True:
                    print ('level: {0}, train_score:{1}, eval_score: {2}'.format(level+1,
                           round(eval_func(labels,self.predict_proba(features)),self.pt_rd),
                           round(eval_func(eval_set[1],self.predict_proba(eval_set[0])),self.pt_rd)))
                else:
                    print ('level: {0}, train_score: {1}'.format(level+1,
                           round(eval_func(labels,self.predict_proba(features)),self.pt_rd)))
        self.upper_limits=(self.features.max()+1).copy()
        self.lower_limits=(self.features.min()-1).copy()
        self.parent_dict={c:p for p in self.root.iter() for c in p}
        if merge:
            merge_pass=[]
            print ('remerging...')
            merge_list=[{'selected_indexes':x.get('selected_indexes'),'nodes':[x.get('node')],'p':np.double(x.get('node').get('p'))} for x in finished_samples_list]
            #merge_list.sort(key=lambda x:len(x.get('selected_indexes')))
            print (len(merge_list))
            break_flag=False
            while (1):
                merge_list.sort(key=lambda x:len(x.get('selected_indexes')),reverse=False)
                for item in [x for x in merge_list if len(x.get('selected_indexes'))==1]:
                    #print (item.get('nodes'))
                    for neighbour in [x for x in merge_list if x!=item]:
                        if self.is_neighbour(neighbour.get('nodes')+item.get('nodes')):
                            N1=len(item.get('selected_indexes'))
                            N2=len(neighbour.get('selected_indexes'))
                            #n1=int(N1*item.get('p')+0.1)
                            #n2=int(N2*neighbour.get('p')+0.1)
                            n1=sum(self.labels[item.get('selected_indexes')])
                            n2=sum(self.labels[neighbour.get('selected_indexes')])
                            loss_l,loss_r=self.expected_gain(n1,N1,n2,N2)
                            if (loss_l+loss_r)<0:
                                #################task decision#############
                                new_p=(n1+n2+1)/(N1+N2+2)
                                new_node={'selected_indexes':item.get('selected_indexes')+neighbour.get('selected_indexes'),
                                          'nodes':item.get('nodes')+neighbour.get('nodes'),'p':new_p}
                                for node_tmp in item.get('nodes'):
                                    node_tmp.set('p',str(new_p))
                                for node_tmp in neighbour.get('nodes'):
                                    node_tmp.set('p',str(new_p))
                                break_flag=True
                                list_tmp=merge_list
                                list_tmp.remove(neighbour)
                                list_tmp.remove(item)
                                list_tmp.append(new_node)
                                merge_pass+=[[n1,N1,n2,N2]]
                                print ('n1:{0},N1:{1},n2:{2},N2:{3}'.format(n1,N1,n2,N2))
                                break
                    if break_flag==True:
                        break
                if break_flag==False:
                    break
                if break_flag==True:
                    merge_list=list_tmp
                    print ('merged_score_train: {0},merged_score_test: {1}'.format(
                           round(eval_func(labels,self.predict_proba(features)),self.pt_rd),
                           round(eval_func(eval_set[1],self.predict_proba(eval_set[0])),self.pt_rd)))
                    break_flag=False                           
        self.tree=ET.ElementTree(self.root)
        self.root=self.tree.getroot()
        return (self.root)
    
    
    def is_neighbour(self,nodes):
        nodes_boundaries=dict()
        for node in nodes:
            node_upper_limits=self.upper_limits.copy()
            node_lower_limits=self.lower_limits.copy()
            tmp_node=node
            while (tmp_node!=self.root.getchildren()[0]):
                parent=self.parent_dict.get(tmp_node)
                split_feature=tmp_node.tag
                split_point=np.float(tmp_node.get('value'))
                if tmp_node.get('flag')=='l':
                    node_upper_limits[split_feature]=min(node_upper_limits[split_feature],split_point)
                else:
                    node_lower_limits[split_feature]=max(node_lower_limits[split_feature],split_point)
                tmp_node=parent
            nodes_boundaries[node]={'upper_limits':node_upper_limits,'lower_limits':node_lower_limits}
        G=nx.Graph()        
        G.add_nodes_from(nodes)
        for node1 in nodes:
            for node2 in [node for node in nodes if node!=node1]:
                node1_upper_limits=nodes_boundaries.get(node1).get('upper_limits')
                node1_lower_limits=nodes_boundaries.get(node1).get('lower_limits')
                node2_upper_limits=nodes_boundaries.get(node2).get('upper_limits')
                node2_lower_limits=nodes_boundaries.get(node2).get('lower_limits')
                boundaries_tmp=list(zip(node1_upper_limits,node1_lower_limits,node2_upper_limits,node2_lower_limits))
                connected_flag=True
                for upper1,lower1,upper2,lower2 in boundaries_tmp:
                    if upper1<lower2 or lower1>upper2:
                        connected_flag=False
                        break
                if connected_flag==True:
                    G.add_edge(node1,node2)
        return(nx.is_connected(G))
        
        
    def greedy_search(self,l_son_index,r_son_index,split_feature,split_point,greedy_node,level,max_greedy_times,N1,N2,
                      expected_info_gain_l,expected_info_gain_r,l_score,r_score):
        node=greedy_node
        self.greedy_N1=N1
        self.greedy_N2=N2
        l_son=ET.SubElement(node,split_feature,{'value':str(split_point),"flag":"l","level":str(level),"p":str(l_score),
                                                "expected_info_gain":str(expected_info_gain_l)})
        r_son=ET.SubElement(node,split_feature,{'value':str(split_point),"flag":"r","level":str(level),"p":str(r_score),
                                                "expected_info_gain":str(expected_info_gain_r)})
        greedy_samples_list=[{'selected_indexes':l_son_index,'node':l_son,'level':level,'coef':(N1+1)/(N1+N2+2),
                              'info_gain':expected_info_gain_l},\
                              {'selected_indexes':r_son_index,'node':r_son,'level':level,'coef':(N2+1)/(N1+N2+2),
                               'info_gain':expected_info_gain_r}]
        for greedy_iter in range(max_greedy_times):
            expected_info_gain_list=[]
            for greedy_selected_samples_info in greedy_samples_list:
                greedy_samples_list_tmp=greedy_samples_list
                greedy_samples_list_tmp.remove(greedy_selected_samples_info)
                selected_samples=greedy_selected_samples_info.get('selected_indexes')
                if selected_samples!=[]:
                    node=greedy_selected_samples_info.get('node')##############
                    level=greedy_selected_samples_info.get('level')
                    coef=greedy_selected_samples_info.get('coef')
                    new_level=level+1
                    selected_labels=self.labels[selected_samples]
                    selected_features=self.features.loc[selected_samples]
                    expected_info_gains=selected_features.apply(lambda col:self.info_gain_cal(col,selected_labels,
                                                                                              self.min_leaf,
                                                                                              self.method,
                                                                                              self.task,                                            
                                                                                              self.min_split),axis=0)
                    split_feature=expected_info_gains.idxmax(axis=1)[0]
                    expected_info_gain,expected_info_gain_l,expected_info_gain_r,split_point,l_score,r_score,N1,N2=tuple(expected_info_gains[split_feature].values)
                    expected_info_gain = expected_info_gain*(self.greedy_N1+self.greedy_N2)/self.total_samples
                    expected_info_gain_l = expected_info_gain_l*(self.greedy_N1+self.greedy_N2)/self.total_samples
                    expected_info_gain_r = expected_info_gain_r*(self.greedy_N1+self.greedy_N2)/self.total_samples
                    l_son_index=selected_features[selected_features[split_feature]<split_point].index.tolist()
                    r_son_index=selected_features[selected_features[split_feature]>=split_point].index.tolist()
                    expected_info_gain_all=sum([tmp.get('coef')*tmp.get('info_gain') for tmp in greedy_samples_list_tmp])
                    expected_info_gain_all+=coef*expected_info_gain_l*(N1+1)/(N1+N2+2)
                    expected_info_gain_all+=coef*expected_info_gain_r*(N2+1)/(N1+N2+2)
                    expected_info_gain_list+=[(expected_info_gain_all,expected_info_gain_l,expected_info_gain_r,
                                               greedy_samples_list_tmp,node,split_feature,split_point,new_level,
                                               l_score,r_score,l_son_index,r_son_index,coef)]
            expected_info_gain_all,expected_info_gain_l,expected_info_gain_r,greedy_samples_list,node,split_feature,split_point,level,l_score,r_score,l_son_index,r_son_index,coef=max(expected_info_gain_list)
            l_son=ET.SubElement(node,split_feature,{'value':str(split_point),"flag":'l','level':str(level),'p':str(l_score),
                                                    'expected_info_gain':str(expected_info_gain_l)})
            r_son=ET.SubElement(node,split_feature,{'value':str(split_point),'flag':'r','level':str(level),'p':str(r_score),
                                                    'expected_info_gain':str(expected_info_gain_r)})
            greedy_samples_list.append({'selected_indexes':l_son_index,'node':l_son,'level':level,'coef':coef*(N1+1)/(N1+N2+2),
                                        'info_gain':expected_info_gain_l})
            greedy_samples_list.append({'selected_indexes':r_son_index,'node':r_son,'level':level,'coef':coef*(N2+1)/(N1+N2+2),
                                        'info_gain':expected_info_gain_r})
            if expected_info_gain_all>self.min_greedy_impurity_increase:
                break
        if expected_info_gain_all<=self.min_greedy_impurity_increase:
            self.remove(greedy_node)
            print('greedy_no_found')
            return('no_positive_found')
        else:
            print ('greedy_gain: {}'.format(round(expected_info_gain_all,self.pt_rd)))
            return([{'selected_indexes':lst.get('selected_indexes'),'node':lst.get('node'),'level':lst.get('level')} for lst in greedy_samples_list])
    def remove(self,node):
        children=node.getchildren()
        for child in children:
            if child.getchildren()==[]:
                node.remove(child)
                return(0)
            else:
                self.remove(child)
            node.remove(child)
        return(0)
        
    def save(self,xmldir):
        self.tree.write(xmldir)
    def load(self,xmldir):
        self.tree=ET.parse(xmldir)
        self.root=self.tree.getroot()
    #def predict(self,features):
    def predict_proba(self,features):
        return features.apply(lambda row:self.decision(self.root,row),axis=1)
            
    def decision(self,root,feature):
        root_node=root.getchildren()[0]
        children=root_node.getchildren()
        if children==[]:
            return root_node.get('p')
        while(children!=[]):
            child=children[0]
            split_feature=child.tag
            split_point=child.get('value')
            if np.float(feature[split_feature])<np.float(split_point):
                flag='l'
            else:
                flag='r'
            for child in children:
                if (child.get('flag')=='l') and (flag=='l'):
                    new_node=child
                if (child.get('flag')=='r') and (flag=='r'):
                    new_node=child
            node=new_node
            children=node.getchildren()
        return np.float(node.get('p'))
'''
from sklearn.model_selection import train_test_split

data=pd.read_csv('./data/wine/winequality-white.csv',sep=';')
data['labels']=data['quality'].apply(lambda x:0 if int(x)<6 else 1)
#data['labels']=data['quality']
features=data.drop(['quality','labels'],axis=1)
features.columns=map(lambda x:'feature'+str(x),list(range(len(features.columns))))
labels=data['labels'].values

x_train,x_test,y_train,y_test=train_test_split(features,labels,test_size=0.2,random_state=100)
x_train=x_train.reset_index(drop=True)
clf=prior_tree()
#tree=clf.fit(x_train,y_train,method='normal',task='regression',max_depth=-1,evaluation=True,eval_set=[x_test,y_test],eval_func=lambda x,y:np.std(x-np.array(y)),greedy=False,max_greedy_times=10)
#tree=clf.fit(x_train,y_train,method='prior',task='classification',min_info_gain=0,min_leaf=1,max_depth=-1,evaluation=True,eval_set=[x_test,y_test],eval_func=roc_auc_score,greedy=True,max_greedy_times=10)
tree=clf.fit(x_train,y_train,method='prior',task='classification',min_info_gain=0,merge=False,min_leaf=1,max_depth=-1,
             prior_num = 1, prior_den =2, priord_num=1,
             evaluation=True,eval_set=[x_test,y_test],eval_func=log_loss,greedy=False)
xmldir='../data/wine/model_prior.xml'
clf.save(xmldir)
pred=clf.predict_proba(x_test)
#from sklearn.metrics import roc_auc_score
#print('ROC score: {0}'.format(round(roc_auc_score(y_test,pred),5)))
'''
'''
########CROSS_VALIDATION#################
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
x_tmp,x_test,y_tmp,y_test=train_test_split(features,labels,test_size=0.2,random_state=10)
kf=KFold(n_splits=4)
model_list=[]
pred_eval_list=[]
pred_list=[]
x_tmp=x_tmp.reset_index(drop=True)
for train,evaluation in kf.split(y_tmp):
    x_train=x_tmp.loc[train,:]
    y_train=y_tmp[train]
    x_eval=x_tmp.loc[evaluation,:]
    y_eval=y_tmp[evaluation]
    x_train=x_train.reset_index(drop=True)
    clf=prior_tree()
   # clf=DecisionTreeClassifier(criterion='entropy',max_depth=25,min_samples_split=100,min_samples_leaf=1,min_impurity_decrease=0)
    model=clf.fit(x_train,y_train,method='prior',min_info_gain=0,min_leaf=1,max_depth=-1)
    #model=clf.fit(x_train,y_train)
    model_list+=[model]
    pred_eval=pd.DataFrame(clf.predict_proba(x_eval)).iloc[:,1].values
 #   pred_eval=clf.predict_proba(x_eval)
    pred_score=roc_auc_score(y_eval,pred_eval)
    pred_eval_list+=[pred_score]
    print ('AUC score: {0}'.format(pred_score))
 #   pred_test=clf.predict_proba(x_test)
    pred_test=pd.DataFrame(clf.predict_proba(x_test)).iloc[:,1].values
    pred_list+=[pred_test]
print (pred_eval_list)
print (roc_auc_score(y_test,pd.DataFrame(pred_list).mean(0)))
print (np.mean(pred_eval_list))
'''