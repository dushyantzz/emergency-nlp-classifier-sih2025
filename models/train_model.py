import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizer,
    TFDistilBertForSequenceClassification,
    TFTrainingArguments,
    TFTrainer
)
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


class EmergencyClassifierTrainer:
    """Trainer class for emergency classification model using TensorFlow"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
        # Set random seeds for reproducibility
        tf.random.set_seed(config['seed'])
        np.random.seed(config['seed'])
        os.environ['PYTHONHASHSEED'] = str(config['seed'])
        
        # Check for GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"💻 Using GPU: {gpus[0]}")
            try:
                # Enable memory growth
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
        else:
            print("💻 Using CPU")
        
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
        
        # Tokenize and prepare datasets
        train_encodings = self.tokenizer(
            train_texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='tf'
        )
        
        val_encodings = self.tokenizer(
            val_texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='tf'
        )
        
        test_encodings = self.tokenizer(
            test_texts,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='tf'
        )
        
        # Create TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': train_encodings['input_ids'],
                'attention_mask': train_encodings['attention_mask']
            },
            np.array(train_labels, dtype=np.int32)
        )).batch(self.config['batch_size']).shuffle(1000).prefetch(tf.data.AUTOTUNE)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': val_encodings['input_ids'],
                'attention_mask': val_encodings['attention_mask']
            },
            np.array(val_labels, dtype=np.int32)
        )).batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        test_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': test_encodings['input_ids'],
                'attention_mask': test_encodings['attention_mask']
            },
            np.array(test_labels, dtype=np.int32)
        )).batch(self.config['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        return train_dataset, val_dataset, test_dataset, test_labels
    
    def create_model(self):
        """Create DistilBERT model for classification"""
        print(f"\n🤖 Creating model: {self.config['model_name']}")
        
        model = TFDistilBertForSequenceClassification.from_pretrained(
            self.config['model_name'],
            num_labels=len(self.label_map)
        )
        
        # Compile model
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')]
        
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        # Count parameters
        total_params = sum([tf.size(w).numpy() for w in model.trainable_variables])
        print(f"Model parameters: {total_params:,}")
        print(f"Trainable parameters: {total_params:,}")
        
        return model
    
    def train(self, train_dataset, val_dataset):
        """Train the model"""
        print(f"\n🚀 Starting training...")
        
        # Create model
        model = self.create_model()
        
        # Create callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=f"{self.config['output_dir']}/checkpoints/checkpoint-{{epoch:02d}}",
                save_weights_only=True,
                monitor='val_accuracy',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Calculate steps per epoch
        steps_per_epoch = len(train_dataset)
        validation_steps = len(val_dataset)
        
        # Train model
        history = model.fit(
            train_dataset,
            epochs=self.config['num_epochs'],
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        print(f"\n💾 Saving model to {self.config['output_dir']}...")
        os.makedirs(self.config['output_dir'], exist_ok=True)
        model.save_pretrained(self.config['output_dir'])
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
        
        return model, history
    
    def evaluate(self, model, test_dataset, test_labels):
        """Evaluate model on test set"""
        print("\n📊 Evaluating on test set...")
        
        # Get predictions
        predictions = model.predict(test_dataset, verbose=1)
        pred_labels = np.argmax(predictions.logits, axis=1)
        
        # Calculate metrics
        print("\n" + "="*80)
        print("CLASSIFICATION REPORT")
        print("="*80)
        print(classification_report(
            test_labels,
            pred_labels,
            target_names=list(self.label_map.keys()),
            digits=4
        ))
        
        # Confusion matrix
        print("\n" + "="*80)
        print("CONFUSION MATRIX")
        print("="*80)
        cm = confusion_matrix(test_labels, pred_labels)
        print("\nRows: True labels, Columns: Predicted labels")
        print(f"Categories: {list(self.label_map.keys())}\n")
        print(cm)
        
        # Overall accuracy
        accuracy = (pred_labels == test_labels).mean()
        print(f"\n🎯 Overall Test Accuracy: {accuracy*100:.2f}%")
        
        return accuracy


def main():
    """Main execution"""
    print("🔥 Emergency Classifier Training - SIH 2025 (TensorFlow)")
    print("="*80)
    
    # Initialize trainer
    trainer_obj = EmergencyClassifierTrainer()
    
    # Load data
    train_dataset, val_dataset, test_dataset, test_labels = trainer_obj.load_data()
    
    # Train model
    model, history = trainer_obj.train(train_dataset, val_dataset)
    
    # Evaluate
    accuracy = trainer_obj.evaluate(model, test_dataset, test_labels)
    
    print("\n" + "="*80)
    print("✨ Training pipeline complete!")
    print(f"🎯 Final test accuracy: {accuracy*100:.2f}%")
    print("\n👉 Next step: Run 'python models/convert_to_tflite.py' to convert to TFLite")
    print("="*80)


if __name__ == "__main__":
    main()
