from base_memory import *
import networkx as nx
import random
import matplotlib.pyplot as plt
import pickle
from abc import ABC, abstractmethod


class KnowledgeGraphMemory(Memory):
    def __init__(self, text,**kwargs):
        pass
    
    @abstractmethod
    def create(self, text):
        pass

    @abstractmethod
    def sample(self, query):
        pass

    @abstractmethod
    def visualize(self):
        pass


#print(nx.__version__)
""" kg = KnowledgeGraphMemory()
kg.create(sample_text)
kg.sample(query=None)
kg.visualize() """

