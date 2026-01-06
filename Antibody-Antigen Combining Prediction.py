import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBList, PDBParser, Polypeptide
from abnumber import Chain as AbChain
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

class AntibodyBindingPredictor:
    def __init__(self):
        # 物理化学的性質の定義
        self.aa_properties = {
            'ALA': [1.80, 6.00, 89.1], 'ARG': [-4.50, 10.76, 174.2], 'ASN': [-3.50, 5.41, 132.1],
            'ASP': [-3.50, 2.77, 133.1], 'CYS': [2.50, 5.07, 121.2], 'GLN': [-3.50, 5.65, 146.1],
            'GLU': [-3.50, 3.22, 147.1], 'GLY': [-0.40, 5.97, 75.1], 'HIS': [-3.20, 7.59, 155.2],
            'ILE': [4.50, 6.02, 131.2], 'LEU': [3.80, 5.98, 131.2], 'LYS': [-3.90, 9.74, 146.2],
            'MET': [1.90, 5.74, 149.2], 'PHE': [2.80, 5.48, 165.2], 'PRO': [-1.60, 6.30, 115.1],
            'SER': [-0.80, 5.68, 105.1], 'THR': [-0.70, 5.60, 119.1], 'TRP': [-0.90, 5.89, 204.2],
            'TYR': [-1.30, 5.66, 181.2], 'VAL': [4.20, 5.96, 117.1]
        }
        self.model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        self.le_res = LabelEncoder()
        self.le_region = LabelEncoder()
        self.is_trained = False

    def _get_features(self, chain_id, sequence):
        """配列から物理化学的・構造的特徴量を抽出"""
        ab_chain = AbChain(sequence, scheme='chothia')
        features = []
        for pos in ab_chain:
            res_3 = Polypeptide.one_to_three(pos.aa)
            region = ab_chain.get_region(pos)
            props = self.aa_properties.get(res_3, [0, 7, 120])
            features.append({
                'Residue': res_3,
                'Region': region,
                'Is_CDR': 1 if 'CDR' in region else 0,
                'Hydrophobicity': props[0],
                'pI': props[1],
                'MW': props[2],
                'Std_Pos': pos.string
            })
        return pd.DataFrame(features)

    def train(self, training_df):
        """学習データ(DataFrame)を用いたモデル訓練"""
        print("Training model...")
        X = training_df[['Residue_Idx', 'Region_Idx', 'Is_CDR', 'Hydrophobicity', 'pI', 'MW']]
        y = training_df['Label']
        self.model.fit(X, y)
        self.is_trained = True
        print("Training complete.")

    def predict(self, sequence):
        """未知配列の結合部位予測"""
        if not self.is_trained:
            raise Exception("Model is not trained yet.")
        
        df_feat = self._get_features('H', sequence) # 簡易的に重鎖として処理
        # エンコード
        df_feat['Residue_Idx'] = self.le_res.transform(df_feat['Residue'])
        df_feat['Region_Idx'] = self.le_region.transform(df_feat['Region'])
        
        X = df_feat[['Residue_Idx', 'Region_Idx', 'Is_CDR', 'Hydrophobicity', 'pI', 'MW']]
        probs = self.model.predict_proba(X)[:, 1]
        df_feat['Binding_Prob'] = probs
        return df_feat

    def save_model(self, path='antibody_model.pkl'):
        joblib.dump({'model': self.model, 'le_res': self.le_res, 'le_region': self.le_region}, path)

    def load_model(self, path='antibody_model.pkl'):
        data = joblib.load(path)
        self.model = data['model']
        self.le_res = data['le_res']
        self.le_region = data['le_region']
        self.is_trained = True