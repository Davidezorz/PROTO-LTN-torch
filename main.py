import data
from model import *

import pytorch_lightning as pl
from lightning_model import ZSLLightningModel

import utils
from config import config



def main():
    print("main running")
    config_file = config()

    device = utils.get_device('cuda') 
    print(f"device: {device}")

    print(config_file.hidden_dense_sizes)
    data_module = data.ZSLDataModule(base_path='./data', 
                                     dataset_name='AWA2', 
                                     batch_size=32,
                                     use_raw_images=config_file.compute_feature)
    data_module.prepare_data()
    data_module.setup(stage='fit')
    data_module.setup(stage='validate')
    data_module.data_summary()

    for i in range(0, 0):
        data.verify_embeddings_match(idx_to_check=i)


    # Define model
    model = ZSLLightningModel(config_file, data_module.all_data)
    model.to(device)


    # 3. Train
    trainer = pl.Trainer(max_epochs=config_file.epochs, accelerator=device)
    trainer.fit(model, data_module)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")