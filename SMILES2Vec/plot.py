import matplotlib.pyplot as plt

# Load the log file
file_path = 'cleaned_test.log'  # Replace with the path to your log file
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract Val R² values and their corresponding epochs
val_r2_values = []
train_r2_values = []
epochs = []
epoch = 0
for line in lines:
    if "Val R²:" in line:
        epoch += 1
        parts = line.split(",")
        for part in parts:
            if "Epoch" in part:
                epoch = int(part.split(" ")[-1].strip())
            if "Val R²:" in part:
                val_r2 = float(part.split(":")[-1].strip())
            if "Train R²:" in part:
                train_r2 = float(part.split(":")[-1].strip())
        if epoch < 130:
            epochs.append(epoch)
            val_r2_values.append(val_r2)
            train_r2_values.append(train_r2)

# Plot Train R² and Val R² vs Epoch for epochs < 130
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_r2_values, label='Train R²')
plt.plot(epochs, val_r2_values, label='Val R²')
plt.title('Train R² and Val R² vs Epoch (Epoch < 130)', fontsize=16)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('R²', fontsize=14)
plt.legend(fontsize=12)
plt.ylim(-1, 1)
plt.grid(True)
plt.tight_layout()
plt.savefig('train_val_r2_epoch_plot.png')  # Saves the plot locally
