from time import sleep
import numpy as np
import json

class Function:
    def __init__(self, labels: list[str], name: str):
        self.labels: list[str] = labels
        self.name = name

    def __str__(self):
        return [f"{l}," for l in self.labels]
    
    def get_first_label(self):
        if len(self.labels) == 0:
            return ''
        return self.labels[0]
    
    def get_labels(self):
        return self.labels
    
    def set_label_experimental(self, label):
        self.labels = [label]
    
    def get_name(self):
        return self.name
    
    def __str__(self):
        return json.dumps({
            "name": self.name,
            "labels": self.labels
        })
