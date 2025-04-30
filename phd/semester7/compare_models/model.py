import lightning as L
from torch import optim, nn, utils, Tensor


class CimatModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x = x.view(x.size(0), -1)
        # print(x.shape)
        z = self.model(x)
        loss = nn.functional.binary_cross_entropy(z, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
