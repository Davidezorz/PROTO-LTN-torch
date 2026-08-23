import torch

def get_device(device: str = None) -> str:                                       
    """Selects the best available device or verifies the requested one.
       If device is None: CUDA -> MPS -> CPU"""      
    if (device in [None, 'cuda']) and torch.cuda.is_available():                #   ╭ Device auto
        return 'cuda'                                                           # ◀─┤ detection  
    if (device in [None, 'mps']) and torch.backends.mps.is_available():         #   │
        return 'mps'                                                            #   │
    if device not in [None, 'cpu']:                                             #   │
        print("From get_device function: only 'cpu' is avaible")                #   ╰
    return 'cpu'
    