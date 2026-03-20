import math
import wandb
import time, gc
import copy
import torch
import wandb
from tqdm import tqdm
from collections import defaultdict
from utils.utils import create_path
from utils.utils import EarlyStopping, AverageMeter, InstantMeter

class SSLMAE_Learner:
    def __init__(
        self, 
        epochs,
        model, 
        optimizer,
        fabric,
        lr=1e-3,
        patience=7,
        warmup_epochs=5,
        n_accumulate=5,
        min_lr=1e-5,
        mode='supervised',
        save_model_path="saved_models",
    ):
        super().__init__()
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.fabric = fabric
        self.lr = lr
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.mode = mode
        self.save_model_path = save_model_path
        self.n_accumulate = n_accumulate        
        self.best_loss = 1e3
        self.best_epoch = 0
        self.last_model_path = None
        self.global_step = 0
        
        create_path(self.save_model_path)
        self.early_stopping = EarlyStopping(patience=self.patience)

        if self.fabric.global_rank == 0:
            self.info_message("Model directory: {}", self.save_model_path)

    def adjust_learning_rate(self, optimizer, epoch):
        """ Decay the learning rate with half-cycle cosine after warmup
            credits:
            https://github.com/facebookresearch/mae/blob/main/util/lr_sched.py
            https://github.com/megvii-research/FullMatch/blob/master/fullmatch.py
        """
        if epoch < self.warmup_epochs:
            lr = self.lr * epoch / self.warmup_epochs 
        else:
            lr = self.min_lr + (self.lr - self.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - self.warmup_epochs) / (self.epochs - self.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

    def save_model(self, epoch, save_path, loss, model_state_dict):
        #self.fabric.save(save_path, model_state_dict)
        torch.save(
            {   'epoch': epoch + 1,
                "state_dict": model_state_dict,
                'loss': loss,
            },
            save_path,
        )
    
    def fit(self, dataloaders, logger = True):    
        """Train the model across specified epochs, handling both labeled and unlabeled data if required."""
        self.logger = logger

        if logger:
            if self.fabric.global_rank == 0:
                wandb.init(
                project="SSL-MAE-lightning",
                name = self.mode,
                config={
                    "epochs": self.epochs,
                })
                wandb.watch(self.model, log_freq=1)
            
        gc.collect()
        training_history = defaultdict(list)

        if torch.cuda.is_available():
            self.info_message("[INFO] Using GPU: {}", torch.cuda.get_device_name())

        self.info_message("Training for {} epochs...", self.epochs)
        start_time = time.time()

        for epoch in range(self.epochs):
            if self.mode == 'semi_supervised':
                train_loss, _ = self.train_one_epoch_u_l(epoch, dataloaders)
            elif self.mode == 'supervised':
                train_loss, _ = self.train_one_epoch(epoch, dataloaders['labeled'])
            elif self.mode == 'unlabeled_only':
                train_loss, _ = self.train_one_epoch(epoch, dataloaders['unlabeled'])
            else:
                raise ValueError(
                    f"Unsupported training mode: {self.mode}. Choose from {['semi_supervised','supervised','unlabeled_only']}")

            valid_loss, _ = self.valid_one_epoch(epoch, dataloaders['val'])

            self.early_stopping(valid_loss)

            if valid_loss < self.best_loss:
                best_model_state_dict = copy.deepcopy(self.model.state_dict())
                self.info_message(
                    "Validation loss improved from {:.4f} to {:.4f}. Saving model...",
                    self.best_loss, valid_loss
                )
                self.best_loss = valid_loss
                self.best_epoch = epoch
                self.last_model_path = f"{self.save_model_path}/model_epoch_{self.best_epoch + 1}-loss_{self.best_loss:.4f}.pth"

            training_history['Train Loss'].append(train_loss)
            training_history['Valid Loss'].append(valid_loss)

            if self.early_stopping.early_stop:
                self.info_message("Early stopping at epoch: {}", epoch + 1)
                break

        training_time = time.time() - start_time
        self.info_message("Training complete in {:.0f}h {:.0f}m {:.0f}s",
                        training_time // 3600, (training_time % 3600) // 60, training_time % 60)


        if best_model_state_dict and self.fabric.global_rank == 0:
            self.save_model(self.best_epoch, self.last_model_path, self.best_loss, best_model_state_dict)

        #self.fabric.barrier()

        # Clean up to free memory
        del self.model, self.optimizer
        torch.cuda.empty_cache()
        gc.collect()

        if logger:
            logger.finish()

        return training_history
    
    def train_one_epoch_u_l(self, epoch, dataloaders):  
        t = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()    
        curr_lr = InstantMeter('', '')
        end = time.time()

        p_bar = tqdm(
            enumerate(zip(dataloaders['labeled'], dataloaders['unlabeled'])),
            total=len(dataloaders['unlabeled']),
            desc=f"Epoch {epoch + 1} - Train Step: ", position=0, leave=True
                )

        for step, batches in p_bar:
            batch_x, batch_u = batches
            self.global_step = epoch * len(dataloaders['unlabeled']) + step
        
            x, mask_x, non_masked, y = batch_x # labeled batch
            u, mask_u, _, _ = batch_u # unlabeled batch

            x = torch.cat((x, u), dim=0)
            mask = torch.cat((mask_x, mask_u), dim=0)

            # Accumulate gradient self.n_accumulate batches at a time
            is_accumulating = step % self.n_accumulate != 0
            # we use a per iteration (instead of per epoch) lr scheduler
            if step % self.n_accumulate == 0:
                self.adjust_learning_rate(self.optimizer, step / len(dataloaders['unlabeled']) + epoch)

            # gradient accumulation if applied
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating): 
                # generate prediction
                outputs = self.model(x, mask, non_masked, y) # on all data
                # compute the joing loss
                loss = outputs['loss']
                w = outputs['w']
                # update losses
                losses.update(loss.item(), x.size(0))  
                self.fabric.backward(loss)   

            # Gradient accumulation #
            if not is_accumulating:
                self.optimizer.step()
                self.optimizer.zero_grad() 

            if self.logger:
                if self.fabric.global_rank == 0:
                    wandb.log({
                        "learning_rate/lr": curr_lr.val, 
                        "loss/train_loss_step": loss, 
                        "global_step": self.global_step
                        }, commit=True)
                                
            curr_lr.update(self.optimizer.param_groups[0]['lr'])
            batch_time.update(time.time() - end)
            end = time.time()


            p_bar.set_postfix(LR = curr_lr, loss=losses.avg, bt=batch_time.avg, w = w)   

        if self.logger is not None:
            if self.fabric.global_rank == 0:
                wandb.log({"loss/train_loss_epoch": losses.avg, "epoch": epoch}, commit=True)

        gc.collect()
        torch.cuda.empty_cache()
        
        return losses.avg, int(time.time() - t)

    def train_one_epoch(self, epoch, dataloader):  
        t = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter()    
        curr_lr = InstantMeter('', '')
        end = time.time()       

        p_bar = tqdm(enumerate(
            dataloader), total=len(dataloader), 
            desc=f"Epoch {epoch + 1} - Train Step: ", position=0, leave=True
            )
        
        for step, batch in p_bar:
            self.global_step = epoch * len(dataloader) + step
            x, mask, non_masked, y = batch

            # Accumulate gradient self.n_accumulate batches at a time
            is_accumulating = step % self.n_accumulate != 0
            # we use a per iteration (instead of per epoch) lr scheduler
            if step % self.n_accumulate == 0:
                self.adjust_learning_rate(self.optimizer, step / len(dataloader) + epoch)

            # gradient accumulation if applied
            with self.fabric.no_backward_sync(self.model, enabled=is_accumulating): 
                # generate prediction
                outputs = self.model(x, mask, non_masked, y) # on all data
                # compute the joing loss
                loss = outputs['loss']
                w = outputs['w']
                # update losses
                losses.update(loss.item(), x.size(0))  
                self.fabric.backward(loss)   

            # Gradient accumulation #
            if not is_accumulating:
                self.optimizer.step()
                self.optimizer.zero_grad() 

            if self.logger:
                if self.fabric.global_rank == 0:
                    wandb.log({
                        "learning_rate/lr": curr_lr.val, 
                        "loss/train_loss_step": loss, 
                        "global_step": self.global_step
                        }, commit=True)
        
            curr_lr.update(self.optimizer.param_groups[0]['lr'])
            batch_time.update(time.time() - end)
            end = time.time()


            p_bar.set_postfix(
                LR = curr_lr,
                loss=losses.avg,
                bt=batch_time.avg,
                w = w,
                )   

        if self.logger is not None:
            if self.fabric.global_rank == 0:
                wandb.log({"loss/train_loss_epoch": losses.avg, "epoch": epoch}, commit=True)
                                
        gc.collect()
        torch.cuda.empty_cache()
        return losses.avg, int(time.time() - t)

    @torch.inference_mode()
    def valid_one_epoch(self, epoch, val_loader):
        self.model.eval()
          
        t = time.time()
        batch_time = AverageMeter()
        losses = AverageMeter() 
        end = time.time()

        p_bar = tqdm(enumerate(val_loader), f"Epoch {epoch + 1} - Val Step: ", position=0, leave=True)
        for step, data in p_bar:    
            x, mask, non_masked, y = data
            # generate prediction
            outputs = self.model(x, mask, non_masked, y)
            # compute the joing loss
            loss = outputs['loss']
            test_loss = self.fabric.all_gather(loss).mean()
            w = outputs['w']
            # update losses
            losses.update(test_loss.item(), x.size(0))
            if self.logger:
                if self.fabric.global_rank == 0:
                    wandb.log({"loss/val_loss_step": loss, "global_step": self.global_step}, commit=True)
            
            self.global_step += 1

        if self.logger is not None:
            if self.fabric.global_rank == 0:
                wandb.log({"loss/val_loss_epoch": losses.avg, "epoch": epoch}, commit=True)

        batch_time.update(time.time() - end)
        end = time.time()


        p_bar.set_postfix(loss=losses.avg, bt=batch_time.avg, w = w)      

        gc.collect()
        p_bar.close() 
        return losses.avg, int(time.time() - t)

    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)
