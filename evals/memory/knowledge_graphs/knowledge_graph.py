from ..base_memory import *
from abc import abstractmethod
import networkx as nx
import os
import pickle
import matplotlib.pyplot as plt

class KnowledgeGraphMemory(Memory):
    def __init__(self, text,**kwargs):
        self.graph = nx.Graph()
        pass
    
    @abstractmethod
    def create(self, **kwargs):
        pass

    
    def sample_semantic_similarity(self, query):
        pass

    def sample_ontology(self, query):
        pass

    def sample(self, query):
        keywords = query.split()
        context = ''        # naive approach to just add context
        related_triplets = set()
        for keyword in keywords:
            related_triplets.update(self.find_related_triplets(keyword))
        for triplet in related_triplets:
            context += triplet[0] + ' is related by: ' + triplet[1] + ', to entity: ' + triplet[2] + ', '
    
        return context
    
    def find_related_triplets(self, keyword):
        related_triplets = []
        for edge in self.graph.edges(data=True):
            if keyword.lower() in edge[0].lower() or keyword.lower() in edge[1].lower():
                related_triplets.append((edge[0], edge[2]['relation'], edge[1]))
        return related_triplets
    
    def visualize(self):
        plt.figure(figsize=(10,10))

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='seagreen', alpha=0.9,
                labels={node: node for node in self.graph.nodes()})
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=nx.get_edge_attributes(self.graph, 'relation'))
        plt.axis('off')
        plt.show()

    def save_graph(self, directory, filename):
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(os.path.join(directory, filename), 'wb') as f:
            pickle.dump(self.graph, f)
    
    def load_graph(self, directory, filename):
        if not os.path.exists(directory):
            print("directory does not exist")
            raise Exception("directory does not exist")
        with open(os.path.join(directory, filename), 'rb') as f:
            self.graph = pickle.load(f)