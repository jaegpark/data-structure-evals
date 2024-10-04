from .knowledge_graph import *
from transformers import BertForTokenClassification, BertTokenizer
from nltk.tag import StanfordNERTagger
from dotenv import load_dotenv
from tqdm import tqdm

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import spacy
import os
import torch
import openai
import pickle

# We need to use the tokenizer manually since we need special tokens.
test_text = """
Napoleon Bonaparte[a] (born Napoleone Buonaparte; 15 August 1769 – 5 May 1821), later known by his regnal name Napoleon I,[b] was a Corsican-born French military commander and political leader who rose to prominence during the French Revolution and led successful campaigns during the Revolutionary Wars. He was the de facto leader of the French Republic as First Consul from 1799 to 1804, then Emperor of the French from 1804 until 1814 and again in 1815. Napoleon's political and cultural legacy endures to this day, as a highly celebrated and controversial leader. He initiated many liberal reforms that have persisted in society, and is considered one of the greatest military commanders in history. His campaigns are still studied at military academies worldwide. Between three and six million civilians and soldiers died in what became known as the Napoleonic Wars.[2][3]
"""

test2 = """
Napoleon was born on the island of Corsica, not long after its annexation by France, to a native family descending from minor Italian nobility.[4][5] He supported the French Revolution in 1789 while serving in the French army, and tried to spread its ideals to his native Corsica. He rose rapidly in the Army after he saved the governing French Directory by firing on royalist insurgents. In 1796, he began a military campaign against the Austrians and their Italian allies, scoring decisive victories and becoming a national hero. Two years later, he led a military expedition to Egypt that served as a springboard to political power. He engineered a coup in November 1799 and became First Consul of the Republic.

Differences with the United Kingdom meant France faced the War of the Third Coalition by 1805. Napoleon shattered this coalition with victories in the Ulm campaign, and at the Battle of Austerlitz, which led to the dissolution of the Holy Roman Empire. In 1806, the Fourth Coalition took up arms against him. Napoleon defeated Prussia at the battles of Jena and Auerstedt, marched the Grande Armée into Eastern Europe, and defeated the Russians in June 1807 at Friedland, forcing the defeated nations of the Fourth Coalition to accept the Treaties of Tilsit. Two years later, the Austrians challenged the French again during the War of the Fifth Coalition, but Napoleon solidified his grip over Europe after triumphing at the Battle of Wagram.

Hoping to extend the Continental System, his embargo against Britain, Napoleon invaded the Iberian Peninsula and declared his brother Joseph the King of Spain in 1808. The Spanish and the Portuguese revolted in the Peninsular War aided by a British army, culminating in defeat for Napoleon's marshals. Napoleon launched an invasion of Russia in the summer of 1812. The resulting campaign witnessed the catastrophic retreat of Napoleon's Grande Armée. In 1813, Prussia and Austria joined Russian forces in a Sixth Coalition against France, resulting in a large coalition army defeating Napoleon at the Battle of Leipzig. The coalition invaded France and captured Paris, forcing Napoleon to abdicate in April 1814. He was exiled to the island of Elba, between Corsica and Italy. In France, the Bourbons were restored to power.

Napoleon escaped in February 1815 and took control of France.[6] The Allies responded by forming a Seventh Coalition, which defeated Napoleon at the Battle of Waterloo in June 1815. The British exiled him to the remote island of Saint Helena in the Atlantic, where he died in 1821 at the age of 51.

Napoleon had an extensive impact on the modern world, bringing liberal reforms to the lands he conquered, especially the regions of the Low Countries, Switzerland and parts of modern Italy and Germany. He implemented many liberal policies in France and Western Europe.[c]
"""
class StagedKG(KnowledgeGraphMemory):

    def __init__(self, text, NER, RE, load=False, **kwargs):
        self.text = text
        self.ner_method = NER
        self.re_method = RE
        self.ner_entities = []
        self.graph = None
        self.load = load

        if not self.load:
            # Stage 1: Find entities
            if self.ner_method == "Spacy":
                self.spacy_ner()
            elif self.ner_method == "BERT":
                self.bert_ner()
            elif self.ner_method == "CoreNLP":
                self.core_nlp_ner()

            print ("Finished NER")
            print ("NER ENTITIES: ", self.ner_entities)
            # Stage 2: Find relations between entities
            if self.re_method == "GPT4":
                self.gpt4_relation_extraction()
            elif self.re_method == "GPT3":
                self.gpt3_relation_extraction()
            
            # Stage 3: Construct graph
            print ("Finished RE")
            print ("RELATIONS: ", self.relations)

            self.create()
            self.save_graph("evals/memory/extractions/graphs/", "end_to_end_graph")
            print ("Saved Graph")
        else:
            self.load_graph("evals/memory/extractions/graphs/", "end_to_end_graph")
            self.visualize()
            #print (self.graph.get_edge_data("Napoleon", "the French Republic")["relation"])
            print (self.graph.nodes())

            edge_data = self.graph.get_edge_data("Napoleon", "the French Republic")

            if edge_data is not None:
                if 'relation' in edge_data:
                    print(edge_data["relation"])
                else:
                    print("'relation' not found in edge data")
            else:
                print("There is no edge between 'Napoleon' and 'the French Republic'")

    # Helper functions
    @staticmethod
    def group_entities(tokens, labels):
        """Group BERT tokens into entities."""
        entities = []
        entity_tokens = []
        current_label = None
        for token, label in zip(tokens, labels):
            if label.startswith('B') or label == 'O':
                if entity_tokens:
                    entities.append((' '.join(entity_tokens), current_label))
                    entity_tokens = []
                current_label = label
            if label != 'O':
                entity_tokens.append(token)
        if entity_tokens:
            entities.append((' '.join(entity_tokens), current_label))
        return entities
    
    def combine_subwords(self, entities):
        combined_entities = []
        current_entity = ""
        current_label = ""

        for entity, label in entities:
            if entity.startswith("##"):
                current_entity += entity[2:]
            else:
                if current_entity:
                    combined_entities.append((current_entity, current_label))
                current_entity = entity
                current_label = label

        # Add last entity if it exists
        if current_entity:
            combined_entities.append((current_entity, current_label))

        return combined_entities

    # NER methods
    def spacy_ner(self):
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(self.text)
        self.ner_entities = [(ent.text, ent.label_) for ent in doc.ents]

    def bert_ner(self):
        bert_model = BertForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")
        bert_tokenizer = BertTokenizer.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english")

        inputs = bert_tokenizer(self.text, return_tensors="pt")
        outputs = bert_model(**inputs)
        predictions = torch.argmax(outputs.logits, dim=2)
        predicted_labels = [bert_model.config.id2label[id] for id in predictions[0].numpy()]
        tokens = bert_tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        entities = [(token, label) for token, label in zip(tokens, predicted_labels) if "B-" in label or "I-" in label]

        # ensure that each tuple has a recognized label
        self.ner_entities = [(entity, label.split('-')[1] if '-' in label else label) for entity, label in entities]
        # optional step to combine subwords into single entities
        # self.ner_entities = self.combine_subwords(entities)

    def core_nlp_ner(self):
        stanford_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    '..', '..', '..', 'data', 'stanford-ner-4.2.0', 'stanford-ner-2020-11-17')
        stanford_ner_file = os.path.join(stanford_dir, 'classifiers', 'english.all.3class.distsim.crf.ser.gz')
        stanford_jar_file = os.path.join(stanford_dir, 'stanford-ner.jar')
        
        stanford_ner_tagger = StanfordNERTagger(stanford_ner_file, stanford_jar_file, encoding='utf-8')

        words = self.text.split()
        self.ner_entities = stanford_ner_tagger.tag(words)

    # RE methods
    def gpt3_relation_extraction(self):
        # Prompt format: "{entity1} is related to {entity2} by"
        from itertools import combinations
        entity_pairs = list(combinations(self.ner_entities, 2))  # considering all possible pairs of entities
        
        load_dotenv('../../.env.local')  
        openai.api_key = os.getenv('OPENAI_API_KEY')

        relations = []
        for (entity1, _), (entity2, _) in tqdm(entity_pairs):
            prompt = f"Answer in one sentence: {entity1} is related to {entity2} by"
            response = openai.Completion.create(engine="text-ada-001", prompt=prompt, max_tokens=60)
            relation_text = response.choices[0].text.strip()
            relations.append((entity1, entity2, relation_text))  # creating a triple

        self.relations = relations


    # Parent Graph methods
    def create(self):
        self.graph = nx.Graph()

        for entity, label in self.ner_entities:
            self.graph.add_node(entity, label=label)

        for entity1, entity2, relation in self.relations:
            self.graph.add_edge(entity1, entity2, relation=relation)
    
    
kg = StagedKG(text=test_text, NER="Spacy", RE="GPT3", load=True)
