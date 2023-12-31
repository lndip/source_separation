import sys
from torch import cuda
from pathlib import Path

from mask_unet_train import train_masking_network
from mask_unet_test import test_masking_network
from utils import MASKING_NET_DIR, get_files_from_dir_with_pathlib

def main():
    # Check if CUDA is available, else use CPU
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(f'Process on {device}', end='\n\n')

    job_idx = int(sys.argv[1])

    epoch_train_loss, epoch_val_loss, network_path = train_masking_network(batch_size=64,
                                                                            patience=20,
                                                                            job_idx=job_idx,
                                                                            device=device,
                                                                            epochs=500)

    network_path = Path(MASKING_NET_DIR, f"#{job_idx}", "best_mask.pt")
    
    test_masking_network(device, network_path, batch_size=1, result_dir_name= Path(f"{network_path.stem}"), mask_threshold=None)


if __name__ == "__main__":
    main()

    

