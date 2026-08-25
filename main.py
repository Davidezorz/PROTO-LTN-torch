import data
from model import *

import pytorch_lightning as pl
from lightning_model import ZSLLightningModel

import utils
from config import config
import TCAV
import gcam

# ssh -l "Il Giudice" 100.106.7.48
# Documents\.pyvenv\Scripts\activate.bat


def main():
    print("main running")
    config_file = config()
    pl.seed_everything(0)

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
    steps = len(data_module.train_dataloader())
    if config_file.checkpoint:
        model = ZSLLightningModel.load_from_checkpoint(
            checkpoint_path=config_file.checkpoint,
            config_file=config_file,
            all_data=data_module.all_data,
            train_cnn=config_file.train_cnn,
            steps_per_epoch=steps,
            strict=False
        )
    else:
        model = ZSLLightningModel(
            config_file, 
            data_module.all_data, 
            train_cnn=config_file.train_cnn,
            steps_per_epoch=steps
        )
        
    model.to(device)


    #TCAV
    # TCAV.run_tcav(model, data_module, target_class_name="tiger", attribute_name="stripes")
    # TCAV.run_tutorial_tcav_lib(model, data_module, target_class_name="zebra")

    # grad cam
    # gcam.plot_gcam(model, data_module, device)

    # TSNE
    # from visualization import plot_tsne_with_synthetic_vectors
    # plot_tsne_with_synthetic_vectors(model, data_module)
    

    # 3. Train
    trainer = pl.Trainer(max_epochs=config_file.epochs, accelerator=device)
    trainer.fit(model, data_module)





if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt")