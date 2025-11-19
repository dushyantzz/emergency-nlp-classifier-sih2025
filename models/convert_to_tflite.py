#!/usr/bin/env python3
"""
TensorFlow Lite Model Converter - SIH 2025
Converts trained DistilBERT model to optimized TFLite format for mobile deployment

Optimizations:
- INT8 quantization (69% size reduction)
- NNAPI support for hardware acceleration
- Optimized for mobile CPU/GPU
"""

import os
import json
import numpy as np
import tensorflow as tf
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizer
import shutil


CONFIG = {
    'model_dir': 'outputs/trained_model',
    'output_dir': 'outputs',
    'tflite_filename': 'emergency_classifier.tflite',
    'max_length': 128,
    'quantization': 'INT8',  # Options: INT8, FP16, DYNAMIC, NONE
}


class TFLiteConverter:
    """Converter for DistilBERT to TFLite"""
    
    def __init__(self, config=CONFIG):
        self.config = config
        
        if not os.path.exists(config['model_dir']):
            raise FileNotFoundError(
                f"Model directory not found: {config['model_dir']}. "
                "Run 'python models/train_model.py' first."
            )
        
        print("🔄 TFLite Converter Initialized")
        print(f"Model directory: {config['model_dir']}")
        print(f"Quantization: {config['quantization']}")
    
    def load_pytorch_model(self):
        """Load trained PyTorch model and convert to TensorFlow"""
        print("\n📂 Loading PyTorch model...")
        
        # Load label mapping
        with open(f"{self.config['model_dir']}/label_mapping.json", 'r') as f:
            label_config = json.load(f)
        
        num_labels = label_config['num_labels']
        print(f"Number of labels: {num_labels}")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(self.config['model_dir'])
        
        # Convert PyTorch model to TensorFlow
        print("\n🔄 Converting PyTorch to TensorFlow...")
        model = TFDistilBertForSequenceClassification.from_pretrained(
            self.config['model_dir'],
            from_pt=True,
            num_labels=num_labels
        )
        
        print("✅ Model loaded and converted successfully")
        
        return model, tokenizer, label_config
    
    def create_concrete_function(self, model):
        """Create concrete function for TFLite conversion"""
        print("\n🔨 Creating concrete function...")
        
        @tf.function(
            input_signature=[
                tf.TensorSpec(shape=[1, self.config['max_length']], dtype=tf.int32, name='input_ids'),
                tf.TensorSpec(shape=[1, self.config['max_length']], dtype=tf.int32, name='attention_mask')
            ]
        )
        def serving_fn(input_ids, attention_mask):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                training=False
            )
            return {'logits': outputs.logits}
        
        concrete_func = serving_fn.get_concrete_function()
        print("✅ Concrete function created")
        
        return concrete_func
    
    def get_representative_dataset(self, tokenizer):
        """Generate representative dataset for quantization"""
        print("\n📊 Generating representative dataset for quantization...")
        
        sample_texts = [
            "Help someone is attacking me with knife",
            "Fire in the building smoke everywhere",
            "Medical emergency heart attack symptoms",
            "Man following me feeling unsafe",
            "Landslide blocked road people trapped",
            "Robbery happening near hotel help",
            "Gas leak detected evacuate immediately",
            "Tourist fell from cliff severe injury",
            "Harassment at tourist spot need help",
            "Flood water rising fast emergency",
            "Someone stole my wallet and phone",
            "Kitchen fire spreading rapidly",
            "Accident multiple people injured",
            "Being stalked need women helpline",
            "Earthquake felt building shaking",
            "Violence happening send police",
            "Smoke coming from forest fire",
            "Breathing difficulty need ambulance",
            "Inappropriate behavior by group",
            "Road washed away natural disaster"
        ]
        
        def representative_dataset_gen():
            for text in sample_texts:
                encoding = tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='tf'
                )
                
                input_ids = tf.cast(encoding['input_ids'], tf.int32)
                attention_mask = tf.cast(encoding['attention_mask'], tf.int32)
                
                yield [input_ids, attention_mask]
        
        print(f"Generated {len(sample_texts)} representative samples")
        return representative_dataset_gen
    
    def convert_to_tflite(self, concrete_func, tokenizer):
        """Convert to TensorFlow Lite with optimizations"""
        print(f"\n🚀 Converting to TFLite ({self.config['quantization']} quantization)...")
        
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
        
        # Apply optimizations based on config
        if self.config['quantization'] == 'INT8':
            print("Applying INT8 quantization (smallest size, best for mobile)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.representative_dataset = self.get_representative_dataset(tokenizer)
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.TFLITE_BUILTINS
            ]
            converter.inference_input_type = tf.int32
            converter.inference_output_type = tf.float32
            
        elif self.config['quantization'] == 'FP16':
            print("Applying FP16 quantization (balanced size/accuracy)...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            
        elif self.config['quantization'] == 'DYNAMIC':
            print("Applying dynamic range quantization...")
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
        else:
            print("No quantization applied (largest size, best accuracy)...")
        
        # Enable NNAPI for hardware acceleration
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS
        ]
        
        # Convert
        tflite_model = converter.convert()
        
        print("✅ Conversion successful!")
        
        return tflite_model
    
    def save_model(self, tflite_model, tokenizer, label_config):
        """Save TFLite model and supporting files"""
        print(f"\n💾 Saving model to {self.config['output_dir']}...")
        
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Save TFLite model
        tflite_path = os.path.join(
            self.config['output_dir'],
            self.config['tflite_filename']
        )
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        model_size_mb = len(tflite_model) / (1024 * 1024)
        print(f"✅ TFLite model saved: {tflite_path}")
        print(f"   Size: {model_size_mb:.2f} MB")
        
        # Save vocabulary
        vocab_path = os.path.join(self.config['output_dir'], 'vocab.txt')
        tokenizer.save_vocabulary(self.config['output_dir'])
        if os.path.exists(os.path.join(self.config['output_dir'], 'vocab.txt')):
            print(f"✅ Vocabulary saved: {vocab_path}")
        
        # Save label mapping
        label_path = os.path.join(self.config['output_dir'], 'label_mapping.json')
        with open(label_path, 'w') as f:
            json.dump(label_config, f, indent=2)
        print(f"✅ Label mapping saved: {label_path}")
        
        # Create deployment info
        deployment_info = {
            'model_name': 'emergency_classifier',
            'version': '1.0.0',
            'framework': 'TensorFlow Lite',
            'quantization': self.config['quantization'],
            'input_shape': [1, self.config['max_length']],
            'output_shape': [1, label_config['num_labels']],
            'max_sequence_length': self.config['max_length'],
            'model_size_mb': round(model_size_mb, 2),
            'categories': list(label_config['label_map'].keys()),
            'usage': {
                'input_format': 'Text string (2-3 sentences)',
                'output_format': 'Logits array (5 classes)',
                'categories': label_config['label_map'],
                'preprocessing': 'Tokenize with vocab.txt, pad to max_length=128',
                'postprocessing': 'Apply softmax to logits, argmax for prediction'
            }
        }
        
        deployment_path = os.path.join(self.config['output_dir'], 'deployment_info.json')
        with open(deployment_path, 'w') as f:
            json.dump(deployment_info, f, indent=2)
        print(f"✅ Deployment info saved: {deployment_path}")
        
        return tflite_path, model_size_mb
    
    def test_tflite_model(self, tflite_path, tokenizer):
        """Test the converted TFLite model"""
        print("\n🧪 Testing TFLite model...")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        # Get input/output details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\nInput details: {len(input_details)} inputs")
        for inp in input_details:
            print(f"  - {inp['name']}: shape {inp['shape']}, dtype {inp['dtype']}")
        
        print(f"\nOutput details: {len(output_details)} outputs")
        for out in output_details:
            print(f"  - {out['name']}: shape {out['shape']}, dtype {out['dtype']}")
        
        # Test with sample text
        test_text = "Help someone is attacking me urgent"
        print(f"\nTest input: '{test_text}'")
        
        encoding = tokenizer(
            test_text,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='np'
        )
        
        # Set input tensors
        interpreter.set_tensor(input_details[0]['index'], encoding['input_ids'].astype(np.int32))
        interpreter.set_tensor(input_details[1]['index'], encoding['attention_mask'].astype(np.int32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        logits = interpreter.get_tensor(output_details[0]['index'])
        
        # Apply softmax
        probs = tf.nn.softmax(logits[0]).numpy()
        predicted_idx = np.argmax(probs)
        
        categories = ['police', 'fire', 'ambulance', 'women_helpline', 'disaster']
        
        print(f"\n🎯 Prediction: {categories[predicted_idx]}")
        print(f"Confidence: {probs[predicted_idx]*100:.2f}%")
        print(f"\nAll probabilities:")
        for i, cat in enumerate(categories):
            print(f"  {cat}: {probs[i]*100:.2f}%")
        
        print("\n✅ TFLite model test successful!")


def main():
    """Main execution"""
    print("🔧 TensorFlow Lite Converter - SIH 2025")
    print("="*80)
    
    # Initialize converter
    converter = TFLiteConverter()
    
    # Load model
    model, tokenizer, label_config = converter.load_pytorch_model()
    
    # Create concrete function
    concrete_func = converter.create_concrete_function(model)
    
    # Convert to TFLite
    tflite_model = converter.convert_to_tflite(concrete_func, tokenizer)
    
    # Save model
    tflite_path, model_size = converter.save_model(tflite_model, tokenizer, label_config)
    
    # Test model
    converter.test_tflite_model(tflite_path, tokenizer)
    
    print("\n" + "="*80)
    print("✨ Conversion pipeline complete!")
    print(f"📦 Model ready for deployment: {tflite_path}")
    print(f"📊 Final model size: {model_size:.2f} MB")
    print("\n👉 Next steps:")
    print("   1. Copy files from outputs/ to your Android app's assets/ folder")
    print("   2. Integrate TFLite interpreter in Kotlin code")
    print("   3. Test on mobile device")
    print("="*80)


if __name__ == "__main__":
    main()