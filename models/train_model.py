#!/usr/bin/env python3
"""
Emergency Classifier Model Training - SIH 2025
Fine-tunes DistilBERT for multi-class emergency classification

Author: Team Daredevils - AI/ML Developer
"""

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset
import json
from tqdm import tqdm


# Configuration
CONFIG = {
    'model_name': 'distilbert-base-uncased',
    'max_length': 128,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 500,
    'weight_decay': 0.01,
    'output_dir': 'outputs/trained_model',
    'logging_steps': 100,
    'save_steps': 500,
    'eval_steps': 500,
    'seed': 42
}


class EmergencyDataset(Dataset):
    """Custom Dataset for emergency classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class EmergencyClassifierTrainer:
    """Trainer class for emergency classification model"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"💻 Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])
        
        # Label mapping
        self.label_map = {
            'police': 0,
            'fire': 1,
            'ambulance': 2,
            'women_helpline': 3,
            'disaster': 4
        }
        self.id_to_label = {v: k for k, v in self.label_map.items()}
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    
    def load_data(self, filepath='data/emergency_dataset.csv'):
        """Load and prepare dataset"""
        print(f"\n📂 Loading dataset from {filepath}...")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Dataset not found at {filepath}. "
                "Run 'python data/generate_dataset.py' first."
            )
        
        df = pd.read_csv(filepath)
        print(f"Total examples: {len(df)}")
        print(f"\nLabel distribution:")
        print(df['label'].value_counts())
        
        # Encode labels
        df['label_id'] = df['label'].map(self.label_map)
        
        # Split dataset: 80% train, 10% val, 10% test
        train_texts, temp_texts, train_labels, temp_labels = train_test_split(
            df['text'].tolist(),
            df['label_id'].tolist(),
            test_size=0.2,
            random_state=self.config['seed'],
            stratify=df['label_id']
        )
        
        val_texts, test_texts, val_labels, test_labels = train_test_split(
            temp_texts,
            temp_labels,
            test_size=0.5,
            random_state=self.config['seed'],
            stratify=temp_labels
        )
        
        print(f"\n📊 Split sizes:")
        print(f"  Train: {len(train_texts)}")
        print(f"  Validation: {len(val_texts)}")
        print(f"  Test: {len(test_texts)}")
        
        # Create datasets
        train_dataset = EmergencyDataset(
            train_texts, train_labels, self.tokenizer, self.config['max_length']
        )
        val_dataset = EmergencyDataset(
            val_texts, val_labels, self.tokenizer, self.config['max_length']
        )
        test_dataset = EmergencyDataset(
            test_texts, test_labels, self.tokenizer, self.config['max_length']
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_model(self):
        """Create DistilBERT model for classification"""
        print(f"\n🤖 Creating model: {self.config['model_name']}")
        
        model = DistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=len(self.label_map),
            problem_type="single_label_classification"
        )
        
        model.to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        return model
    
    def compute_metrics(self, eval_pred):
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Accuracy
        accuracy = (predictions == labels).mean()
        
        return {'accuracy': accuracy}
    
    def train(self, train_dataset, val_dataset):
        """Train the model"""
        print(f"\n🚀 Starting training...")
        
        # Create model
        model = self.create_model()
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config['output_dir'],
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config['warmup_steps'],
            weight_decay=self.config['weight_decay'],
            logging_dir=f"{self.config['output_dir']}/logs",
            logging_steps=self.config['logging_steps'],
            eval_strategy="steps",
            eval_steps=self.config['eval_steps'],
            save_strategy="steps",
            save_steps=self.config['save_steps'],
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            save_total_limit=2,
            seed=self.config['seed'],
            report_to="none"  # Disable wandb/tensorboard
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save final model
        print(f"\n💾 Saving model to {self.config['output_dir']}...")
        trainer.save_model(self.config['output_dir'])
        self.tokenizer.save_pretrained(self.config['output_dir'])
        
        # Save label mapping
        label_config = {
            'label_map': self.label_map,
            'id_to_label': self.id_to_label,
            'num_labels': len(self.label_map)
        }
        
        with open(f"{self.config['output_dir']}/label_mapping.json", 'w') as f:
            json.dump(label_config, f, indent=2)
        
        print("✅ Training complete!")
        
        return trainer
    
    def evaluate(self, trainer, test_dataset):
        """Evaluate model on test set"""
        print("\n📊 Evaluating on test set...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Calculate metrics
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            true_labels,
            pred_labels,
            target_names=list(self.label_map.keys()),
            digits=4
        ))
        
        # Confusion matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(true_labels, pred_labels)
        print("\nRows: True labels, Columns: Predicted labels")
        print(f"Categories: {list(self.label_map.keys())}\n")
        print(cm)
        
        # Overall accuracy
        accuracy = (pred_labels == true_labels).mean()
        print(f"\n🎯 Overall Test Accuracy: {accuracy*100:.2f}%")
        
        return accuracy


def main():
    """Main execution"""
    print("🔥 Emergency Classifier Training - SIH 2025")
    print("="*80)
    
    # Initialize trainer
    trainer_obj = EmergencyClassifierTrainer()
    
    # Load data
    train_dataset, val_dataset, test_dataset = trainer_obj.load_data()
    
    # Train model
    trainer = trainer_obj.train(train_dataset, val_dataset)
    
    # Evaluate
    accuracy = trainer_obj.evaluate(trainer, test_dataset)
    
    print("\n" + "="*80)
    print("✨ Training pipeline complete!")
    print(f"🎯 Final test accuracy: {accuracy*100:.2f}%")
    print("\n👉 Next step: Run 'python models/convert_to_tflite.py' to convert to TFLite")
    print("="*80)


if __name__ == "__main__":
    main()