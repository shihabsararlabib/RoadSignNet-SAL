import matplotlib.pyplot as plt
import json

# Load V4 training log
def load_json_lines(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]

v4_log = load_json_lines('outputs/logs/optimized_v4_training.json')
v5_log = load_json_lines('outputs/logs/optimized_v5_training.json')

v4_epochs = [entry['epoch'] for entry in v4_log]
v4_train_loss = [entry['train_loss'] for entry in v4_log]
v4_val_loss = [entry['val_loss'] for entry in v4_log]

v5_epochs = [entry['epoch'] for entry in v5_log]
v5_train_loss = [entry['train_loss'] for entry in v5_log]
v5_val_loss = [entry['val_loss'] for entry in v5_log]

plt.figure(figsize=(10,6))
plt.plot(v4_epochs, v4_train_loss, label='V4 Train Loss', color='blue', linestyle='--')
plt.plot(v4_epochs, v4_val_loss, label='V4 Val Loss', color='blue')
plt.plot(v5_epochs, v5_train_loss, label='V5 Train Loss', color='red', linestyle='--')
plt.plot(v5_epochs, v5_val_loss, label='V5 Val Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss: V4 vs V5')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('outputs/loss_comparison_v4_v5.png')
plt.show()
