import math
import os
import time

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import openai


class Test:
    @staticmethod
    def create_prompt(txt):
        prompt = f"Dans ce texte: \n\n{txt}\n\n"
        with open(f"prompt.txt") as file:
            prompt += file.read()
        return prompt

    @staticmethod
    def save_response(response, filename):
        with open(f"CAS/{filename}.csv", "w") as file:
            file.write(f"entity_group,word,correct_label\n")
            file.write(response)

    @staticmethod
    def get_api_key(owner):
        df = pd.read_csv("api_keys.csv")
        return df[0][owner]

    @staticmethod
    def test(owner):
        openai.api_key = Test.get_api_key(owner)
        files = Test.get_files("CAS")
        for i, filename in enumerate(files):
            basename, ext = os.path.splitext(filename)
            with open(f"CAS/{filename}") as file:
                txt = file.read()
                prompt = Test.create_prompt(txt)
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=prompt,
                    max_tokens=math.inf,
                    temperature=0
                )
                Test.save_response(response, basename)
                time.sleep(60)

    @staticmethod
    def metrics():
        files = Test.get_files("CORRECT")
        nb_files = len(files)
        true_labels, predicted_labels = [], []
        for i, filename in enumerate(files):
            basename, ext = os.path.splitext(filename)
            df = pd.read_csv(f"CORRECT/{filename}")
            df.dropna(inplace=True)
            true_labels.extend(df['correct_label'].values)
            predicted_labels.extend(df['entity_group'].values)
        precision = precision_score(true_labels, predicted_labels, average='micro', zero_division=0)
        recall = recall_score(true_labels, predicted_labels, average='micro', zero_division=0)
        f1 = f1_score(true_labels, predicted_labels, average='micro', zero_division=0)
        report = classification_report(true_labels, predicted_labels, zero_division=0)
        # Affichage des résultats
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print(f"Report: \n {report}")

    @staticmethod
    def read_corpora(name: str):
        with open(name) as f:
            return f.read()

    @staticmethod
    def get_files(folder_path: str):
        return [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
