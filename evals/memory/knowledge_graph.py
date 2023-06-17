import base_memory
import spacy
import networkx as nx
import random


class KnowledgeGraphMemory(base_memory.Memory):
    def __init__(self, type="NER"):
        self.type = type
        self.graph = nx.Graph()
        self.nlp = spacy.load("en_core_web_sm")

    def create(self, text):
        doc = self.nlp(text)
        for entity in doc.ents:
            self.graph.add_node(entity.text, label=entity.label_)
        for entity1 in doc.ents:
            for entity2 in doc.ents:
                if entity1 != entity2:
                    self.graph.add_edge(entity1.text, entity2.text)
        nx.write_gpickle(self.graph, "knowledge_graph.gpickle")

    def sample(self, query):
        subgraph_nodes = random.choices(list(self.graph.nodes), k=10)
        subgraph = self.graph.subgraph(subgraph_nodes)
        nx.write_gpickle(subgraph, "knowledge_subgraph.gpickle")
