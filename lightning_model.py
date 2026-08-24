import pytorch_lightning as pl
import torch
import ltn
from model import EmbeddingModel 
import torchvision.models


class ZSLLightningModel(pl.LightningModule):

    def __init__(self, config_file, all_data, 
                 train_cnn=False, 
                 train_emb=True, 
                 steps_per_epoch=1):
        super().__init__()
        self.config_file = config_file
        self.compute_feature = config_file.compute_feature
        self.train_cnn = train_cnn
        self.train_emb = train_emb
        self.steps_per_epoch = steps_per_epoch
        
        # 1. Initialize the core Logic Tensor Network model
        input_dim = all_data['attributes_class_matrix'].shape[1]
        self.embeddingFunction = EmbeddingModel(config_file, input_dim)

        if config_file.compute_feature == True:
            resnet_weights = torchvision.models.ResNet101_Weights.IMAGENET1K_V1
            resnet = torchvision.models.resnet101(weights=resnet_weights)
            # Remove the final fully connected layer. The output is now [Batch, 2048, 1, 1]
            self.cnn = torch.nn.Sequential(*(list(resnet.children())[:-1]), 
                                             torch.nn.Flatten())

        if not config_file.train_cnn and config_file.compute_feature == True:
            self.cnn.requires_grad_(False)
            self.cnn.eval()

        if not config_file.train_emb:
            self.embeddingFunction.requires_grad_(False)
            self.embeddingFunction.eval()

        # 2. Register global data as "buffers"
        self.register_buffer('attributes_class_matrix', all_data['attributes_class_matrix'])
        self.register_buffer('train_classes',           all_data['train_classes'])
        self.register_buffer('test_unseen_classes',     all_data['test_unseen_classes'])
        self.register_buffer('test_seen_classes',       all_data['test_seen_classes'])
        self.register_buffer('all_classes',             all_data['all_classes'])

        # We will use this to collect predictions during the validation step
        self.validation_step_outputs = [[], []]  # Index 0 is ZSL, Index 1 is GZSL


    def configure_optimizers(self):
        params = []
        lr_lambdas = []
        
        warmup_steps = 2 * self.steps_per_epoch

        # Setup Embedding Function Optimization
        if self.train_emb:
            params.append({'params': self.embeddingFunction.parameters(), 'lr': self.config_file.learning_rate})
            
            def emb_warmup(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, self.steps_per_epoch))
                return 1.0
            lr_lambdas.append(emb_warmup)

        # CNN Optimization
        if self.train_cnn:
            params.append({'params': self.cnn.parameters(), 'lr': 1e-5})
            
            def cnn_warmup(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0
            lr_lambdas.append(cnn_warmup)

        # Safety check
        if not params:
            raise ValueError("Both train_emb and train_cnn are False. Nothing to optimize!")

        optimizer = torch.optim.Adam(params)

        # Apply the dynamically built lambdas
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lr_lambdas
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }


    def _process_batch(self, batch):
        data, label, attributes = batch
        if self.compute_feature == True:
            features = self.cnn(data)
        else:
            features = data

        return features, label, attributes


    def training_step(self, batch, batch_idx):
        train_feature, train_label, _ = self._process_batch(batch)
        ltn.device = self.device

        # Get prototypes for the seen training classes
        selected_attributes = self.attributes_class_matrix[self.train_classes]
        prototype = self.embeddingFunction(selected_attributes)

        # Calculate Axioms (from model.py)
        axioms_satisfiability, prototype_var = self.embeddingFunction.axioms(
            train_feature, 
            train_label,
            search_space=self.train_classes,
            prototype1  =prototype, 
            config_file =self.config_file
        )

        # Calculate logical loss
        agg_sat = self.embeddingFunction.satisfiabilityAggregation(axioms_satisfiability)
        
        if self.config_file.loss == "1-aggregator":
            loss = 1. - agg_sat
        elif self.config_file.loss == "log":
            loss = -torch.log(agg_sat)

        # Add Regularization
        if self.config_file.regularize:
            l2_loss = sum(torch.sum(param ** 2) / 2.0 for param in self.embeddingFunction.parameters())
            loss += self.config_file.regularization_parameter * l2_loss

        # Calculate Accuracy (without tracking gradients)
        with torch.no_grad():
            train_feature_var = ltn.Variable("train_feature", train_feature)
            truth_values = self.embeddingFunction.isOfClass(train_feature_var, prototype_var).value
            predictions = self.train_classes[torch.argmax(truth_values, dim=-1)]
            acc = torch.mean((predictions == train_label).float())

        # Lightning automatically logs these to TensorBoard/Weights & Biases
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, on_step=False, on_epoch=True)

        return loss


    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        # Because we returned two dataloaders in the DataModule, Lightning passes 
        # dataloader_idx = 0 for ZSL, and dataloader_idx = 1 for GZSL.
        test_feature, test_label, _ = self._process_batch(batch)
        ltn.device = self.device

        # Select the correct search space
        if dataloader_idx == 0:
            search_space = self.test_unseen_classes
        else:
            search_space = self.all_classes

        # Forward pass for testing
        selected_attributes = self.attributes_class_matrix[search_space]
        prototype = self.embeddingFunction(selected_attributes)

        test_feature_var = ltn.Variable("test_feature", test_feature)
        prototype_var = ltn.Variable("prototype", prototype)

        # Calculate truth values and predict the class with the highest truth
        truth_values = self.embeddingFunction.isOfClass(test_feature_var, prototype_var).value
        predictions = search_space[torch.argmax(truth_values, dim=-1)]

        # Save the batch predictions so we can calculate global metrics at the end of the epoch
        self.validation_step_outputs[dataloader_idx].append({
            'preds': predictions,
            'labels': test_label
        })


    def on_validation_epoch_end(self):
        # 1. Evaluate Standard ZSL Metrics (Dataloader 0)
        if len(self.validation_step_outputs[0]) > 0:
            zsl_preds = torch.cat([x['preds'] for x in self.validation_step_outputs[0]])
            zsl_labels = torch.cat([x['labels'] for x in self.validation_step_outputs[0]])
            zsl_acc = torch.mean((zsl_preds == zsl_labels).float())
            self.log('top1_zsl', zsl_acc, prog_bar=True)

        # 2. Evaluate Generalized ZSL (GZSL) Metrics (Dataloader 1)
        if len(self.validation_step_outputs[1]) > 0:
            gzsl_preds = torch.cat([x['preds'] for x in self.validation_step_outputs[1]])
            gzsl_labels = torch.cat([x['labels'] for x in self.validation_step_outputs[1]])

            # Calculate accuracy exclusively on Unseen test images
            unseen_mask = torch.isin(gzsl_labels, self.test_unseen_classes)
            unseen_acc = torch.mean((gzsl_preds[unseen_mask] == gzsl_labels[unseen_mask]).float()) if unseen_mask.any() else 0.0

            # Calculate accuracy exclusively on Seen test images
            seen_mask = torch.isin(gzsl_labels, self.test_seen_classes)
            seen_acc = torch.mean((gzsl_preds[seen_mask] == gzsl_labels[seen_mask]).float()) if seen_mask.any() else 0.0

            # Harmonic Mean
            if unseen_acc + seen_acc > 0:
                h_mean = 2 * (unseen_acc * seen_acc) / (unseen_acc + seen_acc)
            else:
                h_mean = 0.0

            self.log('top1_gzsl_unseen', unseen_acc)
            self.log('top1_gzsl_seen', seen_acc)
            self.log('H_gzsl', h_mean, prog_bar=True)

        # Clear the memory for the next epoch
        self.validation_step_outputs = [[], []]