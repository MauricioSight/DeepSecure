import torch
from torch import nn

class Trainer:
    def __init__(self, config, logger, tracker, model):
        self.logger = logger
        self.tracker = tracker

        self.config = config
        self.model = model
        self.device = self.__get_device()

        self.training_config = config.get('training')
        self.learning_rate = self.training_config.get('learning_rate')
        self.early_stopping_patience = self.training_config.get('early_stopping_patience')
        self.num_epochs = self.training_config.get('num_epochs')
        self.criterion = self.training_config.get('criterion')
        self.best_val_loss = float("inf")

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
        if val_loss < self.best_val_loss:
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
            if batch_idx % 1000 == 0 or batch_idx == len(train_loader) - 1:
                self.logger.info('Epoch: {} \t[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx, len(train_loader.dataset), 
                    100. * batch_idx / len(train_loader), loss.item()))
        
        train_loss = train_loss / len(train_loader)

        return train_loss


    def validate(self, val_loader, criterion_without_reduction = None, epoch = None):
        if criterion_without_reduction is None:
            _, criterion_without_reduction = self.__get_criterion()

        self.model.eval()
        val_loss = 0
        y_pred = torch.tensor([]).to(self.device)
        y_true = torch.tensor([]).to(self.device)
        labels = []

        with torch.no_grad():
            for data, target, label in val_loader:
                out = self.model(data)
                loss = criterion_without_reduction(out, target)
                mean_loss = loss.mean(dim=(1, 2))

                val_loss += mean_loss.mean().item()
                y_pred = torch.cat((y_pred, mean_loss))
                y_true = torch.cat((y_true, target))
                labels.extend(label.tolist())

        val_loss = val_loss / len(val_loader)

        if epoch is not None:
            self.logger.info('Epoch: {} \tValidation Loss: {:.6f}'.format(epoch, val_loss))
        else:
            self.logger.info('Test Loss: {:.6f}'.format(val_loss))

        return y_true, y_pred, labels, val_loss

    def execute(self, train_loader, val_loader):
        criterion, criterion_without_reduction = self.__get_criterion()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.logger.warning(f"Running for {self.num_epochs} epochs")
        self.logger.info(f"-------------------- Training started -------------------")
        for epoch in range(self.num_epochs):
            train_loss = self.train(train_loader, criterion, optimizer, epoch)
            _, _, _, val_loss = self.validate(val_loader, criterion_without_reduction=criterion_without_reduction, 
                                              epoch=epoch)
            ret = self.__check_early_stopping(val_loss)

            self.tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if (ret < 0):
                self.logger.info(
                    f"Early stopping! Validation loss hasn't improved for {self.early_stopping_patience} epochs")
                break
