from knowledge_graph import KnowledgeGraphMemory
from transformers import pipeline, AutoTokenizer, DistilBertForTokenClassification
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class EndToEndKG(KnowledgeGraphMemory):
    """
    Class for an end-to-end construction of a knowledge graph.
    End to End means there is a singular language model that extracts triplets tuples from the text.

    Supported models:
        - Babelscape/rebel-large

    """
    def __init__(self, text,
                  model="Babelscape/rebel-large",
                  tokenizer="Babelscape/rebel-large",
                  pipelines = True,
                  **kwargs):
        self.text = text
        
        if (pipelines):
            self.triplet_extractor = pipeline('text2text-generation', model=model, tokenizer=tokenizer)
            self.extracted_text = self.triplet_extractor.tokenizer.batch_decode([self.triplet_extractor(text, return_tensors=True, return_text=False)[0]["generated_token_ids"]])
        else:
            print("No other pipelines supported yet.")
        
        self.triplets = self.extract_triplets(self.extracted_text[0])
        self.graph = self.create_knowledge_graph(self.triplets)

        print("done constructing the graph.")
        self.visualize_knowledge_graph()

    
    def create_knowledge_graph(self, triplets):
        # Create a directed graph
        G = nx.DiGraph()
   
        # Add nodes and edges to the graph
        for triplet in triplets:
            G.add_node(triplet['head'], value=np.random.rand())  # Add a random value as node attribute
            G.add_node(triplet['tail'], value=np.random.rand())  # Add a random value as node attribute
            G.add_edge(triplet['head'], triplet['tail'], relation=triplet['type'])

        return G


    def find_related_triplets(self, keyword):
        related_triplets = []
        for edge in self.graph.edges(data=True):
            if keyword.lower() in edge[0].lower() or keyword.lower() in edge[1].lower():
                related_triplets.append((edge[0], edge[2]['relation'], edge[1]))
        return related_triplets
    
    def extract_triplets(self, text):
        triplets = []
        relation, subject, relation, object_ = '', '', '', ''
        text = text.strip()
        current = 'x'
        for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
            if token == "<triplet>":
                current = 't'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                    relation = ''
                subject = ''
            elif token == "<subj>":
                current = 's'
                if relation != '':
                    triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
                object_ = ''
            elif token == "<obj>":
                current = 'o'
                relation = ''
            else:
                if current == 't':
                    subject += ' ' + token
                elif current == 's':
                    object_ += ' ' + token
                elif current == 'o':
                    relation += ' ' + token
        if subject != '' and relation != '' and object_ != '':
            triplets.append({'head': subject.strip(), 'type': relation.strip(),'tail': object_.strip()})
        # triplets: list of dicts, each dict has 3 keys: head, type, tail
        return triplets
    
    def visualize(self):
        plt.figure(figsize=(10,10))

        pos = nx.spring_layout(self.graph)
        nx.draw(self.graph, pos, edge_color='black', width=1, linewidths=1,
                node_size=500, node_color='seagreen', alpha=0.9,
                labels={node: node for node in self.graph.nodes()})
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=nx.get_edge_attributes(self.graph, 'relation'))
        plt.axis('off')
        plt.show()

    def create(self, text):
        pass

    def sample(self, query):
        keywords = query.split()
        context = ''        # naive approach to just add context
        related_triplets = set()
        for keyword in keywords:
            related_triplets.update(self.find_related_triplets(keyword))
        for triplet in related_triplets:
            context += triplet['head'] + ' ' + triplet['type'] + ' ' + triplet['tail'] + ' '
    
        return context
    
    
        
    