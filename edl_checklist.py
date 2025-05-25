import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer  # Or your specific model/tokenizer
import logging
from models import EDLHead # Add this line

# 1.  Check Data Loading and Label Consistency
def check_data_and_labels(labels, num_classes):
    print("\n--- 1. Check Data and Labels ---")
    print("Labels shape:", labels.shape)
    unique_labels = torch.unique(labels)
    print("Unique labels:", unique_labels)
    if torch.any(labels < 0) or torch.any(labels >= num_classes):
        print(f"ERROR: Labels contain values outside the range [0, {num_classes-1}]")
        return False
    if unique_labels.shape[0] != num_classes and num_classes>2:
        print(f"ERROR: Number of unique labels ({unique_labels.shape[0]}) does not match num_classes ({num_classes})")
        return False
    print("Labels check passed.")
    return True

# 2.  Check Model Architecture and Forward Pass
def check_model_forward(model, tokenizer, sample_text, num_classes, use_edl):
    print("\n--- 2. Check Model Architecture and Forward Pass ---")
    model.eval()  # Important for consistent behavior
    inputs = tokenizer(sample_text, padding=True, truncation=True, return_tensors="pt").to(model.device) # Move inputs to model's device
    with torch.no_grad():
        output = model(**inputs)  # **inputs to handle dict
    
    if use_edl:
        print("Model output (alpha) shape:", output.shape)
        if len(output.shape) != 2 or output.shape[1] != num_classes:
            print(f"ERROR: EDLHead output should have shape [batch_size, num_classes] = [1, {num_classes}].  Got {output.shape}")
            return False
    else:
        print("Model output (logits) shape:", output.logits.shape)
        if len(output.logits.shape) != 2 or output.logits.shape[1] != num_classes:
            print(f"ERROR: Softmax output should have shape [batch_size, num_classes] =  [1, {num_classes}]. Got {output.logits.shape}")
            return False
    print("Model forward pass check passed.")
    return True


# 3. Check Loss Function
def check_loss_function(output, labels, num_classes, use_edl):
    print("\n--- 3. Check Loss Function ---")
    if use_edl:
        def edl_mse_loss(pred_alpha, target, epoch=1, num_classes_loss=2, coeff=1.0): # Renamed num_classes
            S = torch.sum(pred_alpha, dim=1, keepdim=True)
            one_hot = F.one_hot(target, num_classes_loss).float()
            loss = torch.sum((one_hot - pred_alpha / S) ** 2, dim=1, keepdim=True)
            reg = coeff * torch.sum((pred_alpha - 1) ** 2, dim=1, keepdim=True)
            return torch.mean(loss + reg)
        loss_value = edl_mse_loss(output, labels, num_classes_loss=num_classes)
        print("EDL loss value:", loss_value.item())
    else:
        loss_fn = torch.nn.CrossEntropyLoss()
        loss_value = loss_fn(output, labels)
        print("Softmax loss value:", loss_value.item())
    print("Loss function check passed (value calculated).")
    return True

def main():
    # Configuration (adapt to your setup)
    pretrained_model_name = "roberta-base"  # Or your model
    num_classes = 2  # Important:  Set this correctly
    use_edl = True # Change this to test both branches
    sample_text = ["This is a test sentence."]  # For model forward pass check
    
    #logging
    logging.basicConfig(level=logging.INFO)

    # 0. Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Check Data and Labels
    # Create some dummy label data for the check.  Adapt this to match
    # the structure of your actual labels (e.g., from your Dataset class).
    batch_size = 8
    labels = torch.randint(0, num_classes, (batch_size,)).to(device)  # Example
    if not check_data_and_labels(labels, num_classes):
        return

    # 2. Check Model Architecture and Forward Pass
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(pretrained_model_name) #Generic
    if use_edl:
        # Replace the classifier with EDLHead (adapt to your model's structure)
        in_features = model.config.hidden_size # this works for roberta.
        model.classifier = EDLHead(in_features, num_classes)
    model.to(device)
    if not check_model_forward(model, tokenizer, sample_text, num_classes, use_edl):
        return

    # 3. Check Loss Function
    # Create a dummy output tensor.  This is crucial to get the shape correct.
    if use_edl:
        output_shape = (batch_size, num_classes)  # For EDLHead
    else:
        output_shape = (batch_size, num_classes)  # For Softmax
    output = torch.randn(output_shape).to(device) # initialize with random numbers
    if not check_loss_function(output, labels, num_classes, use_edl):
        return
    
    print("\nAll checks passed ")

if __name__ == "__main__":
    main()

