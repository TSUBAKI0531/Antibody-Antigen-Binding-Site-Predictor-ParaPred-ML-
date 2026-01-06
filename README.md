# Antibody-Antigen Binding Site Predictor (ParaPred-ML)

## 概要
本プロジェクトは、抗体医薬品の研究開発における初期スクリーニングを支援するための、**機械学習を用いた抗体結合部位（パラトープ）予測システム**です。
PDBから取得した三次元構造データに基づき、抗体の相補性決定領域（CDR）の構造的特徴と、アミノ酸の物理化学的性質を統合して学習を行っています。

## 特徴
- **標準番号付けの自動化**: `abnumber` を活用し、PDB IDに依存しない標準的な Chothia/Kabat 番号付けによる解析を実現。
- **マルチパラメトリック解析**: 残基の種類に加え、疎水性、等電点、分子量、およびCDR領域フラグを特徴量として採用。
- **構造未知配列への対応**: ホモロジーモデリングを介さず、一次配列のみから結合部位の確率を高速に予測。
- **実務向け設計**: クラスベースのPythonモジュールにより、既存の創薬パイプラインやWebアプリへの統合が容易。

## 解析ワークフロー


1. **データ取得**: Biopythonを用いたPDB複合体データの自動取得。
2. **前処理**: 
   - 抗体と抗原の接触残基（距離 $4.0\text{\AA}$ 以下）を正解ラベルとして抽出。
   - `abnumber` によるCDR領域の自動同定。
3. **記述子計算**: アミノ酸ごとの物理化学的指標（Kyte-Doolittle等）の結合。
4. **モデル訓練**: RandomForestアルゴリズムを用いた二値分類モデルの構築。
5. **評価・予測**: 未知配列に対するパラトープ存在確率の算出。

## セットアップ
```bash
pip install biopython abnumber rdkit scikit-learn pandas matplotlib seaborn


## 使い方
from antibody_predictor import AntibodyBindingPredictor

# モデルのロード
predictor = AntibodyBindingPredictor()
predictor.load_model('pretrained_model.pkl')

# 未知の重鎖配列から予測
seq = "EVQLVESGGGLVQPGGSLRLSCAASGFTF..."
results = predictor.predict(seq)

# 結合確率が高いトップ10残基を表示
print(results.sort_values('Binding_Prob', ascending=False).head(10))


## 今後の展望
深層学習（Graph Neural Networks）の導入による空間的特徴の更なる活用。
SAbDab（Structural Antibody Database）全件を用いた大規模学習データの構築。


## ライセンス
MIT License

著者
[Masaki Sukeda / TSUBAKI0531] (Ph.D. in Agriculture / Antibody Drug Researcher)
