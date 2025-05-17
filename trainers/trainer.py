import torch
from torch import nn

from metrics.evaluation import get_inference_time, pytorch_compute_model_size_mb
from utils.experiment_io import get_run_dir, load_run_artifacts

class Trainer:
    def __init__(self, config, logger, tracker, model):
        self.logger = logger
        self.tracker = tracker

        self.config = config
        self.model = model
        self.device = self.__get_device()
        self.run_id = config.get('run_id')
        self.run_dir = get_run_dir(self.run_id)
        self.model_dir = self.run_dir / "model.pt"

        self.training_config = config.get('training')
        self.learning_rate = self.training_config.get('learning_rate')
        self.early_stopping_patience = self.training_config.get('early_stopping_patience')
        self.early_stopping_delta = self.training_config.get('early_stopping_delta', 0.001)
        self.num_epochs = self.training_config.get('num_epochs')
        self.criterion = self.training_config.get('criterion')
        self.best_val_loss = float("inf")

        self.loaded = False
        self.checkpoint_epoch = 0
        self.__load_model_state_dict()

    def __save_model_state_dict(self):
        torch.save(self.model.state_dict(), self.model_dir)
    
    def __load_model_state_dict(self):
        if self.model_dir.exists():
            artifacts = load_run_artifacts(self.run_dir)
            self.model.load_state_dict(artifacts['model_state_dict'])
            self.model = self.model.to(self.device)
            self.loaded = True
            self.checkpoint_epoch = artifacts['metrics']['epochs']

    def __get_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def __get_criterion(self):
            # Get this criterion from configuration parameter
            criterion = None
            if (self.criterion == 'binary_cross_entropy'):
                criterion = nn.BCELoss()
            elif (self.criterion == 'categorical_cross_entropy'):
                criterion = nn.CrossEntropyLoss()
            elif (self.criterion == 'mean_squared_error'):
                criterion = nn.MSELoss()
            elif (self.criterion == 'mean_absolute_error'):
                criterion = nn.L1Loss()
            else:
                raise KeyError(f"Selected criterion : {self.criterion} is NOT available!")

            criterion_without_reduction = None
            if (self.criterion == 'mean_squared_error'):
                criterion_without_reduction = nn.MSELoss(reduction='none')
            elif (self.criterion == 'categorical_cross_entropy'):
                criterion_without_reduction = nn.CrossEntropyLoss(reduction='none')
            elif (self.criterion == 'mean_absolute_error'):
                criterion_without_reduction = nn.L1Loss(reduction='none')
            elif (self.criterion == 'binary_cross_entropy'):
                criterion_without_reduction = nn.BCELoss(reduction='none')

            return criterion, criterion_without_reduction

    def __check_early_stopping(self, val_loss) -> int:
        ret = 0
        # Early stopping update
        if val_loss < self.best_val_loss - self.early_stopping_delta:
            self.__save_model_state_dict()
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement = self.epochs_without_improvement + 1

        # Early stopping condition
        if self.epochs_without_improvement >= self.early_stopping_patience:
            ret = -1

        return ret

    def train(self, train_loader, criterion, optimizer, epoch):
        self.model.train()
        train_loss = 0

        self.model = self.model.to(self.device)

        for batch_idx, (data, target, _) in enumerate(train_loader):
            optimizer.zero_grad()
            out = self.model(data)
            loss = criterion(out, target)
            loss.backward()

            train_loss += loss.item()

            optimizer.step()

            # metrics logs
            if batch_idx % 1000 == 0 or  batch_idx * len(data) == len(train_loader.dataset) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch,  batch_idx * len(data), len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss.item()))
        
        train_loss = train_loss / len(train_loader)

        return train_loss


    def validate(self, val_loader, criterion, epoch):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for data, target, _ in val_loader:
                out = self.model(data)
                loss = criterion(out, target)
                val_loss += loss.item()

        val_loss = val_loss / len(val_loader)

        ret = self.__check_early_stopping(val_loss)
        
        self.logger.info('Epoch: {} \tEarlyStopping: {} out of {}. Val loss: {:.6f}'.format(
            epoch, self.epochs_without_improvement, self.early_stopping_patience, val_loss))

        return ret, val_loss


    def test(self, test_loader, criterion_without_reduction = None):
        if criterion_without_reduction is None:
            _, criterion_without_reduction = self.__get_criterion()

        self.model.eval()
        test_loss = 0
        y_pred = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)
        labels = []

        with torch.no_grad():
            for data, target, label in test_loader:
                out = self.model(data)
                loss = criterion_without_reduction(out, target)
                mean_loss = loss.mean(dim=(1, 2))

                test_loss += mean_loss.mean().item()
                y_pred = torch.cat((y_pred, mean_loss))
                y_true = torch.cat((y_true, target))
                labels.extend(label.tolist())

        test_loss = test_loss / len(test_loader)

        self.logger.info('Test Loss: {:.6f}'.format(test_loss))

        # Resource metrics
        self.logger.info('Collecting resource metrics...')
        first_batch = next(iter(test_loader))
        data = first_batch[0]
        dummy_input = torch.randn_like(data)
        cpu_inference_time, gpu_inference_time, mps_inference_time = get_inference_time(self.model, dummy_input)
        model_size_mb = pytorch_compute_model_size_mb(self.model)
        resource_metrics = {
            'cpu_inference_time': cpu_inference_time,
            'gpu_inference_time': gpu_inference_time,
            'mps_inference_time': mps_inference_time,
            'model_size_mb': model_size_mb
        }

        return y_true, y_pred, labels, test_loss, resource_metrics

    def execute(self, train_loader, val_loader):
        criterion, _ = self.__get_criterion()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.logger.warning(f"Running for {self.num_epochs} epochs")
        self.logger.info(f"-------------------- Training started -------------------")
        if self.loaded:
            _, val_loss = self.validate(val_loader, criterion, -1)
        for epoch in range(self.checkpoint_epoch, self.num_epochs):
            self.epochs = epoch
            train_loss = self.train(train_loader, criterion, optimizer, epoch)
            ret, val_loss = self.validate(val_loader, criterion, epoch)

            self.tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if (ret < 0):
                self.logger.info(
                    f"Early stopping! Validation loss hasn't improved for {self.early_stopping_patience} epochs")
                self.__load_model_state_dict()
                break
