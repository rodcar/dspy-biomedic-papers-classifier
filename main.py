#!/usr/bin/env python3
"""
Minimal DSPy Biomedical Classifier - Evaluation Script
Usage: python main.py <csv_file_path>
"""
import sys
import os
from dotenv import load_dotenv
import dspy
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def setup_dspy():
    """Setup DSPy with OpenAI"""
    load_dotenv()
    if not os.getenv('OPENAI_API_KEY'):
        raise ValueError("OPENAI_API_KEY not found. Set it in .env file")
    lm = dspy.LM('openai/gpt-4o-mini', api_key=os.getenv('OPENAI_API_KEY'))
    dspy.configure(lm=lm)

class ClassifyPaper(dspy.Signature):
    """Classify a biomedical paper into medical categories"""
    title = dspy.InputField(desc="Title of the biomedical paper")
    abstract = dspy.InputField(desc="Abstract of the biomedical paper")
    reasoning = dspy.OutputField(desc="Step by step reasoning for classification")
    categories = dspy.OutputField(desc="Python list of medical categories: neurological, cardiovascular, hepatorenal, oncological")

class BiomedicalClassifier(dspy.Module):
    """Biomedical classifier"""
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(ClassifyPaper)
    
    def forward(self, title, abstract):
        return self.predict(title=title, abstract=abstract)

def load_data_and_model(csv_path):
    """Load CSV data and model"""
    df = pd.read_csv(csv_path, delimiter=';')
    model = BiomedicalClassifier()
    return df, model

def predict_batch(df, model):
    """Make predictions for all papers"""
    predictions = []
    categories = ['neurological', 'cardiovascular', 'hepatorenal', 'oncological']
    
    for _, row in df.iterrows():
        try:
            paper_text = f"Title: {row['title']}\nAbstract: {row['abstract']}"
            pred = model(title=row['title'], abstract=row['abstract'])
            
            # Parse categories from prediction
            pred_cats = []
            pred_text = str(pred.categories).lower() if hasattr(pred, 'categories') else str(pred).lower()
            for cat in categories:
                if cat in pred_text:
                    pred_cats.append(cat)
            
            predictions.append('|'.join(pred_cats))
        except Exception as e:
            print(f"Error predicting: {e}")
            predictions.append('')
    
    return predictions

def calculate_metrics(df, predictions):
    """Calculate F1 scores and confusion matrices"""
    categories = ['neurological', 'cardiovascular', 'hepatorenal', 'oncological']
    results = {}
    
    # Convert to binary format
    for cat in categories:
        y_true = df['group'].apply(lambda x: cat in str(x).split('|')).astype(int)
        y_pred = pd.Series(predictions).apply(lambda x: cat in str(x).split('|')).astype(int)
        
        f1 = f1_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred)
        
        results[cat] = {'f1': f1, 'cm': cm, 'y_true': y_true, 'y_pred': y_pred}
    
    # Weighted F1
    all_f1s = [results[cat]['f1'] for cat in categories]
    weights = [results[cat]['y_true'].sum() / len(results[cat]['y_true']) for cat in categories]
    weighted_f1 = sum(w * f1 for w, f1 in zip(weights, all_f1s)) / sum(weights) if sum(weights) > 0 else 0
    
    return results, weighted_f1

def display_results(results, weighted_f1):
    """Display F1 scores and confusion matrices"""
    categories = ['neurological', 'cardiovascular', 'hepatorenal', 'oncological']
    
    print(f"\n🎯 WEIGHTED F1 SCORE: {weighted_f1:.4f}")
    print("\n📊 Per-Category Results:")
    print("-" * 50)
    
    for cat in categories:
        f1 = results[cat]['f1']
        support = results[cat]['y_true'].sum()
        total = len(results[cat]['y_true'])
        print(f"{cat.capitalize():15} | F1: {f1:.4f} | Support: {support}/{total} ({support/total*100:.1f}%)")
    
    # Plot confusion matrices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, cat in enumerate(categories):
        cm = results[cat]['cm']
        f1 = results[cat]['f1']
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'], ax=axes[i])
        axes[i].set_title(f'{cat.capitalize()} - F1: {f1:.3f}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("💾 Confusion matrices saved to confusion_matrices.png")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <csv_file_path>")
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    try:
        print("🔧 Setting up DSPy...")
        setup_dspy()
        
        print("📁 Loading data and model...")
        df, model = load_data_and_model(csv_path)
        print(f"📊 Loaded {len(df)} papers")
        
        print("🔄 Making predictions...")
        predictions = predict_batch(df, model)
        
        print("📈 Calculating metrics...")
        results, weighted_f1 = calculate_metrics(df, predictions)
        
        display_results(results, weighted_f1)
        
        # Save results
        df['group_predicted'] = predictions
        output_path = csv_path.replace('.csv', '_with_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"💾 Results saved to {output_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
