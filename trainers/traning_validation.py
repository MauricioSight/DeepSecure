import random

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from metrics.evaluation import compute_metrics
from trainers.trainer import Trainer

class TrainValidation:
    def __init__(self, config, logger, model, trainer: Trainer, tracker):
        self.config = config
        self.model = model
        self.logger = logger
        self.tracker = tracker
        self.trainer = trainer

        self.training_config = config.get('training')
        self.batch_size = self.training_config.get('batch_size')
        self.train_size = 0.8
        
    def __seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def __divide_data(self, X, labels: pd.DataFrame):
        # Shuffle the data
        idx = np.arange(X.shape[0])

        train_val_idx, test_idx = train_test_split(idx, train_size=self.train_size, random_state=10, shuffle=True)

        # TODO: Check if it is unsupervised
        benign_idx = labels[labels['label'] == 'Normal'].index

        # Filter train_idx with benign_idx
        train_idx = np.array([i for i in train_val_idx if i in benign_idx])
        train_idx, val_idx = train_test_split(train_idx, train_size=self.train_size, random_state=10, shuffle=True)

        self.logger.info(f"Train size: {len(train_idx)}, Validation size: {len(val_idx)}, Test size: {len(test_idx)}")

        return train_idx, val_idx, test_idx

    def __create_loaders(self, X, y, labels):
        def collate_gpu(batch):
            x, t, l = torch.utils.data.dataloader.default_collate(batch)
            
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
            
            return x.to(device=device), t.to(device=device), l
        
        g = torch.Generator()
        g.manual_seed(42)

        train_subsample, val_subsample, test_subsample = self.__divide_data(X, labels)

        # Create DataFrame for each subset
        train_df = pd.DataFrame([labels.iloc[i]['label'] for i in train_subsample], columns=['label'])
        val_df = pd.DataFrame([labels.iloc[i]['label'] for i in val_subsample], columns=['label'])
        test_df = pd.DataFrame([labels.iloc[i]['label'] for i in test_subsample], columns=['label'])

        self.logger.info(f"Train labels: {train_df['label'].value_counts()}")
        self.logger.info(f"Validation labels: {val_df['label'].value_counts()}")
        self.logger.info(f"Test labels: {test_df['label'].value_counts()}")

        del train_df, val_df, test_df

        data = [[X[i], y[i], i] for i in range(X.shape[0])]

        train_loader = torch.utils.data.DataLoader(
                    data,
                    batch_size=self.batch_size,
                    sampler=train_subsample,
                    generator=g,
                    worker_init_fn=self.__seed_worker,
                    collate_fn=collate_gpu)
        
        val_loader = torch.utils.data.DataLoader(
                    data,
                    batch_size=self.batch_size,
                    sampler=val_subsample,
                    generator=g,
                    worker_init_fn=self.__seed_worker,
                    collate_fn=collate_gpu)
        
        test_loader = torch.utils.data.DataLoader(
                    data,
                    batch_size=self.batch_size,
                    sampler=test_subsample,
                    generator=g,
                    worker_init_fn=self.__seed_worker,
                    collate_fn=collate_gpu)
        
        return train_loader, val_loader, test_loader
    
    def execute(self, values, labels):
        X, y = self.model.featuring(values, labels)

        train_loader, val_loader, test_loader = self.__create_loaders(X, y, labels)
        del  X, y

        # Train the model
        self.trainer.execute(train_loader, val_loader)

        # Test the model
        y_true, y_pred, labels_idx, test_loss = self.trainer.test(test_loader)
        y_pred = y_pred.cpu().numpy()
        del y_true

        # Select labels_idx from labels
        label_values = [labels.iloc[i]['label'] for i in labels_idx]
        del labels_idx

        # Compute metrics
        metrics = compute_metrics(label_values, y_pred, logger=self.logger)

        self.tracker.log_metrics({**metrics, 'test_loss': test_loss})

        del train_loader, val_loader, test_loader

        return label_values, y_pred, metrics
    