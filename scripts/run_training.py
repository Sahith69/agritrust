import sys; sys.path.insert(0, '/home/sahith/Projects/Crop'); import config
import os
import matplotlib.pyplot as plt
import pandas as pd
from src.train import train_model

def plot_history(history, crop, model_type):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train')
    plt.plot(epochs, history['val_loss'], label='Val')
    plt.title(f'{crop} {model_type} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_qwk'], label='Train')
    plt.plot(epochs, history['val_qwk'], label='Val')
    plt.title(f'{crop} {model_type} QWK')
    plt.xlabel('Epoch')
    plt.ylabel('QWK')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.REPORTS_DIR, f"{crop}_{model_type}_history.png"))
    plt.close()

def main():
    config.ensure_dirs()
    
    results = []
    
    # Arecanut
    if os.path.exists(config.ARECANUT_SPLIT_CSV):
        print("\n--- Training Arecanut Baseline ---")
        h_ab = train_model('arecanut', config.ARECANUT_SPLIT_CSV, ordinal=False)
        plot_history(h_ab, 'arecanut', 'baseline')
        results.append({'Crop': 'Arecanut', 'Model': 'Baseline (CE)', 'Best Val QWK': h_ab['best_val_qwk']})
        
        print("\n--- Training Arecanut Ordinal (CORN) ---")
        h_ao = train_model('arecanut', config.ARECANUT_SPLIT_CSV, ordinal=True)
        plot_history(h_ao, 'arecanut', 'ordinal')
        results.append({'Crop': 'Arecanut', 'Model': 'Ordinal (CORN)', 'Best Val QWK': h_ao['best_val_qwk']})
    else:
        print(f"Arecanut split CSV not found: {config.ARECANUT_SPLIT_CSV}")
        
    # Paddy
    if os.path.exists(config.PADDY_SPLIT_CSV):
        print("\n--- Training Paddy Baseline ---")
        h_pb = train_model('paddy', config.PADDY_SPLIT_CSV, ordinal=False)
        plot_history(h_pb, 'paddy', 'baseline')
        results.append({'Crop': 'Paddy', 'Model': 'Baseline (CE)', 'Best Val QWK': h_pb['best_val_qwk']})
        
        print("\n--- Training Paddy Ordinal (CORN) ---")
        h_po = train_model('paddy', config.PADDY_SPLIT_CSV, ordinal=True)
        plot_history(h_po, 'paddy', 'ordinal')
        results.append({'Crop': 'Paddy', 'Model': 'Ordinal (CORN)', 'Best Val QWK': h_po['best_val_qwk']})
    else:
        print(f"Paddy split CSV not found: {config.PADDY_SPLIT_CSV}")
        
    if results:
        print("\n=== Final Results ===")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))

if __name__ == "__main__":
    main()
