import numpy as np
import re

class RuleExtractor:
    
    def __init__(self, model) -> None:
        self.model = model.max_depth_dt
        self.feature_names = model.feature_names
        self.class_names = model.class_names
        self.route = {}
    

    def get_route(self):        
        model = self.model
        leaf_ids = model.tree_.children_left == -1
        leaf_node_indicies = np.where(leaf_ids)[0]

        feature = model.tree_.feature
        threshold = model.tree_.threshold
        n_nodes = model.tree_.node_count
        children_left = model.tree_.children_left
        children_right = model.tree_.children_right
        
        route = {}
        
        for leaf_idx in leaf_node_indicies:
            path = []
            node_idx = leaf_idx
            while node_idx != 0:
                parent_idx = -1
                for j in range(n_nodes):
                    if children_left[j] == node_idx or children_right[j] == node_idx:
                        parent_idx = j
                        break
                if parent_idx == -1:
                    break
                if node_idx == children_left[parent_idx]:
                    path.append(f"{self.feature_names[feature[parent_idx]]} <= {threshold[parent_idx]}")
                else:
                    path.append(f"{self.feature_names[feature[parent_idx]]} > {threshold[parent_idx]}")
                node_idx = parent_idx
            path.reverse()
            route[leaf_idx] = path
        
        return route, leaf_node_indicies
    

    def extract_rule(self, segment_num):
        if len(self.route) == 0 or self.model.n_classes_ == 2:
            self.route, self.leaf_node_indicies = self.get_route()

        leaf_node_values = self.model.tree_.value[self.leaf_node_indicies]
        leaf_node_classes = self.model.classes_[np.argmax(leaf_node_values, axis=2)]

        segment_rule_list = []
        leaf_node_class = np.squeeze(leaf_node_classes)
        segment_leaf_node_indicies = self.leaf_node_indicies[np.where(leaf_node_class==segment_num)]
        
        for leaf_idx in segment_leaf_node_indicies:
            segment_rule_list.append(' [AND] '.join(self.route[leaf_idx]))
        rule_str = ' [OR] \n'.join(segment_rule_list)

        rules = re.sub(r'\d+\.\d+', lambda x: str(round(float(x.group()), 3)), rule_str)
        
        return rules