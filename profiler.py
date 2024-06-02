from sklearn.preprocessing import OrdinalEncoder
import pandas  as pd
import re
import numpy as np
import matplotlib.pyplot as plt

class Profiler :

    def __init__(self, survey, survey_label, mac_to_idx):

        self.survey = self.mapping_id(survey, mac_to_idx)
        self.survey_label = survey_label


    def name_prep(self, x):
        x=re.sub('\.','',x) #kt
        x=re.sub('-','',x) #skt
        x=re.sub(':','',x) #uplus
        x=x.lower()
        return x 


    def mapping_id(self, survey, mac_to_idx):
        # Mapping Mac Address to User ID
        survey.loc[:,'Mac_Address'] = survey.loc[:,'Mac_Address'].map(self.name_prep)
        survey.loc[:,'Mac_Address'] = survey.loc[:,'Mac_Address'].map(mac_to_idx)
        return survey
    

    def feature_engineering(self, onehot_cols, ordinal_cols):
        encoder_ordinal = OrdinalEncoder()

        encoded_survey = pd.get_dummies(self.survey, columns=onehot_cols, prefix=onehot_cols)
        encoded = encoder_ordinal.fit_transform(encoded_survey[ordinal_cols]) #ordinal encoding
        encoded_survey[ordinal_cols] = encoded

        encoded_survey.dropna(axis=1,inplace=True)
        encoded_survey.set_index('Mac_Address', inplace=True)

        return encoded_survey

    
    def dt_feature_importance_raw(self, model, ordinal_cols, measure = 'ratio'):
        #중요한 부분. raw input일 때의 feature importtance 추출
        assert measure.lower() in ['ratio','count']
        
        features = model.dt.tree_.feature
        names = model.feature_names
        thresholds = list(map(int,model.dt.tree_.threshold))
        
        node_names = []
        for idx,feature in enumerate(features):
            if feature >= 0:
                name = names[feature]
                if name in ordinal_cols:
                    node_names.append(name + '<=' + str(thresholds[idx]))
                else:
                    node_names.append(name)
        node_names_left = node_names #왼쪽은 그대로
        node_names = node_names_left 

        left_c = model.dt.tree_.children_left
        right_c = model.dt.tree_.children_right
        node_samples = model.dt.tree_.weighted_n_node_samples
        pos_samples = model.dt.tree_.value[:,:,1].squeeze()
        feature_importance = {key : 0 for key in set(node_names)} # pos, neg


    # dt.tree_.feature : 각 노드가 순서대로 어떤 feature인지. 
        for idx,feature in enumerate(features): 
            if feature >= 0: #해당 노드가 있으면
                importance_parent = pos_samples[idx] / node_samples[idx]

                if measure == 'ratio':

                    for lr,child in enumerate([left_c[idx],right_c[idx]]):
                        importance_child = pos_samples[child] / node_samples[child] # pos sample 비중
                        node_importance = node_samples[idx] * (importance_child - importance_parent) # 비중 변화 측정 
                
                        if node_importance > 0 : # 양수일때만 
                            name = names[feature]
                            if name in ordinal_cols: #ordinal
                                if lr==0: # left --> pos                           해당 노드 특성에 대해서는 +              
                                    feature_importance[name+'<='+str(thresholds[idx])] += node_importance    
                                else : #right --> neg 해당 노드 특성에 대해서는 - 
                                    feature_importance[name+'<='+str(thresholds[idx])] -= node_importance
                            
                            else: #onehot
                                if lr==0: # left --> pos                           해당 노드 특성에 대해서는 +              
                                    feature_importance[name] -= node_importance    
                                else : #right --> neg 해당 노드 특성에 대해서는 - 
                                    feature_importance[name] += node_importance
                            

                elif measure == 'count':
                    
                    for lr,child in enumerate([left_c[idx],right_c[idx]]):
                        #어짜피 방향은 항상 같기 때문에
                        node_importance = pos_samples[idx] * pos_samples[child]                       
                        name  = names[feature]
                        if name in ordinal_cols: #ordinal
                            if lr==0: #left --> +
                                    feature_importance[name+'<='+str(thresholds[idx])] += node_importance    
                            else : #right --> -
                                    feature_importance[name+'<='+str(thresholds[idx])] -= node_importance
                            
                        else: #onehot
                            if lr==0: # left --> pos                           해당 노드 특성에 대해서는 +              
                                feature_importance[name] -= node_importance    
                            else : #right --> neg 해당 노드 특성에 대해서는 - 
                                feature_importance[name] += node_importance
                            

        feature_importance = {key:value/node_samples[0] for key,value in feature_importance.items()} 
        
        return feature_importance
    


    def plot_feature_importance(self, feature_importance):
        features = np.array(list(feature_importance.keys()))
        values = np.array(list(feature_importance.values()))
        sorted_idx = np.argsort(np.abs(values))
        top_5 = np.argsort(np.abs(values))[-5:]

        idx  = np.arange(top_5.shape[0])
        plt.barh(idx,values[top_5], align='center')
        plt.yticks(idx,features[top_5])
        plt.tight_layout()

        idx  = np.arange(values.shape[0])
        plt.figure(figsize=(12,44))
        plt.barh(idx,values[sorted_idx], align='center')
        plt.yticks(idx,features[sorted_idx])
        plt.tight_layout()
        plt.show()