from base_memory import *
import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle

class KnowledgeGraphMemory(Memory):
    def __init__(self, text,**kwargs):
        self.text = text

        pass
    def create(self, text):
        doc = self.nlp(text)
        for entity in doc.ents:
            self.graph.add_node(entity.text, label=entity.label_)
        for entity1 in doc.ents:
            for entity2 in doc.ents:
                if entity1 != entity2:
                    if self.graph.has_edge(entity1.text, entity2.text):
                        # we added this one before, just increase the weight by one
                        self.graph[entity1.text][entity2.text]['weight'] += 1
                    else:
                        # new edge. add with weight=1
                        self.graph.add_edge(entity1.text, entity2.text, weight=1)

        with open("knowledge_graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)

    def sample(self, query):
        subgraph_nodes = random.choices(list(self.graph.nodes), k=10)
        subgraph = self.graph.subgraph(subgraph_nodes)
        with open("knowledge_subgraph.pkl", "wb") as f:
            pickle.dump(subgraph, f)

    def visualize(self):
        fig, ax = plt.subplots()
        with open("knowledge_subgraph.pkl", "rb") as f:
            subgraph = pickle.load(f)
        edge_labels = {(u, v): d['weight'] for u, v, d in subgraph.edges(data=True)}
        pos = nx.spring_layout(subgraph)
        nx.draw(subgraph, pos, with_labels=True)
        nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels)
        plt.show()


#print(nx.__version__)
""" kg = KnowledgeGraphMemory()
kg.create(sample_text)
kg.sample(query=None)
kg.visualize() """

