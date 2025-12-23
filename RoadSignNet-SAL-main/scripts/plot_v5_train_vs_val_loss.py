import matplotlib.pyplot as plt
import json

# Load V5 training log
def load_json_lines(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

v5_log = load_json_lines('outputs/logs/optimized_v5_training.json')
v5_epochs = [entry['epoch'] for entry in v5_log]
v5_train_loss = [entry['train_loss'] for entry in v5_log]
v5_val_loss = [entry['val_loss'] for entry in v5_log]

plt.figure(figsize=(10,6))
plt.plot(v5_epochs, v5_train_loss, label='V5 Train Loss', color='red', linestyle='--')
plt.plot(v5_epochs, v5_val_loss, label='V5 Val Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('V5 Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/v5_train_vs_val_loss.png')
plt.show()
