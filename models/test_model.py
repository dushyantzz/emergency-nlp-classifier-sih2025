import os
import json
import numpy as np
import tensorflow as tf
from transformers import DistilBertTokenizer, TFDistilBertForSequenceClassification


class ModelTester:
    """Test emergency classification models"""
    
    def __init__(self):
        self.model_dir = 'outputs/trained_model'
        self.tflite_path = 'outputs/emergency_classifier.tflite'
        
        # Load label mapping
        with open(f"{self.model_dir}/label_mapping.json", 'r') as f:
            label_config = json.load(f)
        self.id_to_label = {int(k): v for k, v in label_config['id_to_label'].items()}
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
        
        print("🧪 Model Tester Initialized")
    
    def test_tensorflow_model(self, text):
        """Test TensorFlow model"""
        # Load model
        model = TFDistilBertForSequenceClassification.from_pretrained(self.model_dir)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='tf'
        )
        
        # Predict
        outputs = model(encoding)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()
        predicted_idx = np.argmax(probs[0])
        
        return {
            'category': self.id_to_label[predicted_idx],
            'confidence': float(probs[0][predicted_idx]),
            'all_probs': {self.id_to_label[i]: float(probs[0][i]) for i in range(len(self.id_to_label))}
        }
    
    def test_tflite_model(self, text):
        """Test TFLite model"""
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=self.tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='np'
        )
        
        # Set input
        interpreter.set_tensor(input_details[0]['index'], encoding['input_ids'].astype(np.int32))
        interpreter.set_tensor(input_details[1]['index'], encoding['attention_mask'].astype(np.int32))
        
        # Run inference
        interpreter.invoke()
        
        # Get output
        logits = interpreter.get_tensor(output_details[0]['index'])
        probs = tf.nn.softmax(logits[0]).numpy()
        predicted_idx = np.argmax(probs)
        
        return {
            'category': self.id_to_label[predicted_idx],
            'confidence': probs[predicted_idx],
            'all_probs': {self.id_to_label[i]: probs[i] for i in range(len(self.id_to_label))}
        }


def main():
    """Main testing"""
    print("🧪 Emergency Classifier Testing - SIH 2025")
    print("="*80)
    
    # Test scenarios
    test_cases = [
        ("Someone following me with knife threatening help", "police"),
        ("Hotel room caught fire smoke everywhere urgent", "fire"),
        ("Tourist fell from cliff severe injury bleeding", "ambulance"),
        ("Man stalking me feeling very unsafe", "women_helpline"),
        ("Landslide blocked road multiple people trapped", "disaster"),
        ("Robbery happening near market send police", "police"),
        ("Gas leak detected evacuate immediately", "fire"),
        ("Heart attack symptoms chest pain emergency", "ambulance"),
        ("Being harassed by group of men", "women_helpline"),
        ("Flood water rising fast need rescue", "disaster")
    ]
    
    tester = ModelTester()
    
    print("\n🔬 Testing TensorFlow Model")
    print("="*80)
    
    correct = 0
    for text, expected in test_cases:
        result = tester.test_tensorflow_model(text)
        match = "✅" if result['category'] == expected else "❌"
        
        print(f"\n{match} Input: {text[:60]}...")
        print(f"   Expected: {expected} | Predicted: {result['category']} ({result['confidence']*100:.1f}%)")
        
        if result['category'] == expected:
            correct += 1
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"\n🎯 TensorFlow Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    
    # Test TFLite
    if os.path.exists(tester.tflite_path):
        print("\n\n🔬 Testing TFLite Model")
        print("="*80)
        
        correct = 0
        for text, expected in test_cases:
            result = tester.test_tflite_model(text)
            match = "✅" if result['category'] == expected else "❌"
            
            print(f"\n{match} Input: {text[:60]}...")
            print(f"   Expected: {expected} | Predicted: {result['category']} ({result['confidence']*100:.1f}%)")
            
            if result['category'] == expected:
                correct += 1
        
        accuracy = (correct / len(test_cases)) * 100
        print(f"\n🎯 TFLite Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    
    print("\n" + "="*80)
    print("✨ Testing complete!")
    print("="*80)


if __name__ == "__main__":
    main()