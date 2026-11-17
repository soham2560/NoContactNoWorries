import os
import json
import argparse
import yaml
from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType, DataLoaderConfiguration
from accelerate.utils import set_seed

from models.model import ContactTransformer, CT_without_segmask
from utils.sequence import ContactDataset

MAX_GPU_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32

class ContactTrainer:
    """
        Trainer class using HuggingFace Accelerate for distributed training of ContactTransformer.
    """
    def __init__(
        self,
        config: Dict[str, Any],
        train_dataset: ContactDataset,
        val_dataset: Dict[str, ContactDataset],
        model: ContactTransformer,
        checkpoint_dir: str,
        log_wandb: bool = True,
        resume_training: bool = False,
        use_stateful_dataloader: bool = False,
    ):
        
        """
            Initialise the trainer.

            config (Dict[str, Any]): Training config.
            train_dataset (ContactDataset): Training dataset.
            val_dataset (Dict[str, ContactDataset]): Validation datasets.
            model (ContactTransformer): Model to train.
            checkpoint_dir (str): Directory to save checkpoints.
            log_wandb (bool): Whether to log to wandb. Defaults to True.
            resume_training (bool): Whether to resume training from a checkpoint. Defaults to False.
            use_stateful_dataloader (bool): Whether to use stateful dataloader. Defaults to False.
        """

        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model = model
        self.checkpoint_dir = checkpoint_dir
        self.log_wandb = log_wandb
        self.resume_training = resume_training
        self.use_stateful_dataloader = use_stateful_dataloader

        # Extract training parameters
        self.train_batch_size = config['batch_size']
        self.train_lr = config['lr']
        self.max_grad_norm = config['max_grad_norm']
        self.num_epochs = config['epochs']
        self.seed = config['seed']

        # Create directory to save checkpoints
        if self.checkpoint_dir is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Configure accelerator
        dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=self.use_stateful_dataloader)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            log_with='wandb' if self.log_wandb else None,
            dataloader_config=dataloader_config,
        )

        # Initialize Weights & Biases (if enabled)
        if self.log_wandb:
            self.accelerator.init_trackers(
                project_name="contact_prediction",
                config=self.config,
                init_kwargs={"wandb": {"entity": self.config.get("wandb_entity")}}
            )

        # Possibly adjust batch size and gradient accumulation steps
        self._setup_batch_size()

        # Create dataloaders
        self._create_dataloaders()

        # Prepare model, optimizer, etc.
        self._setup_training_components()

    def _setup_batch_size(self) -> None:
        """
        Adjust batch size and calculate gradient accumulation steps if necessary.
        """
        self.gradient_accumulation_steps = 1
        if (
            self.train_batch_size > MAX_GPU_BATCH_SIZE
            and self.accelerator.distributed_type != DistributedType.XLA
        ):
            self.gradient_accumulation_steps = self.train_batch_size // MAX_GPU_BATCH_SIZE
            self.train_batch_size = MAX_GPU_BATCH_SIZE

    def _create_dataloaders(self) -> None:
        """
        Create the train and validation DataLoaders from the datasets.
        """
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        # self.val_loader = DataLoader(
        #     self.val_dataset,
        #     batch_size=EVAL_BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=2,
        #     pin_memory=True,
        #     persistent_workers=True,
        # )
        self.cube_loader = DataLoader(
            self.val_dataset['cube'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.dodec_loader = DataLoader(
            self.val_dataset['dodec'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.letter_r_loader = DataLoader(
            self.val_dataset['letter_r'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.pent_loader = DataLoader(
            self.val_dataset['pent'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.star_loader = DataLoader(
            self.val_dataset['star'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.stairs_loader = DataLoader(
            self.val_dataset['stairs'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        self.hex_prism_loader = DataLoader(
            self.val_dataset['hex_prism'],
            batch_size=EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    def _setup_training_components(self) -> None:
        """
        Set up the model, optimizer, loss function, and place them on the accelerator.
        """
        # Set seed for reproducibility
        set_seed(self.seed)

        # Move model to accelerator device
        self.model = self.model.to(self.accelerator.device)

        # Define loss function and optimizer
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_lr)

        # Prepare everything with accelerator
        (
            self.model,
            self.optimizer,
            self.train_loader,
            self.cube_loader,
            self.dodec_loader,
            self.letter_r_loader,
            self.pent_loader,
            self.star_loader,
            self.stairs_loader,
            self.hex_prism_loader,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.train_loader,
            self.cube_loader,
            self.dodec_loader,
            self.letter_r_loader,
            self.pent_loader,
            self.star_loader,
            self.stairs_loader,
            self.hex_prism_loader,
        )

    def train_one_epoch(
        self,
        epoch: int,
        overall_step: int,
        resume_step: Optional[int] = None
    ) -> int:
        """
        Train the model for one epoch.

        Args:
            epoch (int): Current epoch index.
            overall_step (int): Global step across all epochs.
            resume_step (Optional[int], optional): Step from which to resume training. Defaults to None.

        Returns:
            int: Updated overall_step.
        """
        self.model.train()
        total_loss = 0.0

        # If we're resuming training and skipping steps
        if resume_step is not None and resume_step > 0:
            if not self.use_stateful_dataloader:
                active_dataloader = self.accelerator.skip_first_batches(self.train_loader, resume_step)
            else:
                active_dataloader = self.train_loader
            overall_step += resume_step
        else:
            active_dataloader = self.train_loader

        for step, batch in enumerate(active_dataloader):
            image, depth, hand_joint_pose, desired_hand_joint_pose, contacts = (
                batch.image,
                batch.depth,
                batch.hand_joint_pose,
                batch.desired_hand_joint_pose,
                batch.contact.float()
            )
            outputs = self.model(image, depth, hand_joint_pose, desired_hand_joint_pose)
            B, H, sensors = outputs.shape

            loss = self.loss_fn(outputs.view(B * H, sensors), contacts.view(B * H, sensors))
            loss = loss / self.gradient_accumulation_steps  # gradient accumulation

            total_loss += loss.detach().float()
            self.accelerator.backward(loss)

            if (step + 1) % self.gradient_accumulation_steps == 0 or (step + 1) == len(active_dataloader):
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()

            overall_step += 1

        avg_train_loss = total_loss / len(self.train_loader)
        return overall_step, avg_train_loss

    def _evaluate_one_loader(self, name: str, loader: DataLoader) -> Tuple[float, float, float]:
        """
        Evaluate the model on a single validation DataLoader.

        Returns:
            Tuple[float, float, float]: (validation loss, validation accuracy, validation F1)
            F1 is computed per contact site over all timesteps, then averaged across sites.
        """
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        tp = fp = fn = None  # per-sensor counts, summed over the whole loader

        loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

        with torch.inference_mode():
            for batch in loader:
                image, depth, hand_joint_pose, desired_hand_joint_pose, contacts = (
                    batch.image,
                    batch.depth,
                    batch.hand_joint_pose,
                    batch.desired_hand_joint_pose,
                    batch.contact.float()
                )
                outputs = self.model(image, depth, hand_joint_pose, desired_hand_joint_pose)

                probs = torch.sigmoid(outputs)
                predictions = (probs > 0.5).float()

                # For distributed evaluation, gather results across processes
                predictions, outputs, contacts = self.accelerator.gather_for_metrics(
                    (predictions, outputs, contacts)
                )

                # Flatten the leading dimensions (handle possible multi-GPU shapes)
                B_H, sensors = outputs.view(-1, outputs.shape[-1]).shape
                out_flat = outputs.view(B_H, sensors)
                tgt_flat = contacts.view(B_H, sensors)
                pred_flat = predictions.view(B_H, sensors)
                val_loss += loss_fn(
                    out_flat,
                    tgt_flat
                ).item()

                total += tgt_flat.numel()
                correct += (pred_flat == tgt_flat).sum().item()

                batch_tp = (pred_flat * tgt_flat).sum(dim=0)
                batch_fp = (pred_flat * (1 - tgt_flat)).sum(dim=0)
                batch_fn = ((1 - pred_flat) * tgt_flat).sum(dim=0)
                if tp is None:
                    tp, fp, fn = batch_tp, batch_fp, batch_fn
                else:
                    tp, fp, fn = tp + batch_tp, fp + batch_fp, fn + batch_fn

        avg_val_loss = (val_loss / total) if total > 0 else 0.0
        accuracy = (correct / total) if total > 0 else 0.0
        if tp is not None:
            # F1 = 2TP / (2TP + FP + FN) per sensor, 0 when a sensor has no true or predicted contacts
            denom = 2 * tp + fp + fn
            f1_per_sensor = torch.where(denom > 0, 2 * tp / denom.clamp(min=1), torch.zeros_like(denom))
            f1 = f1_per_sensor.mean().item()
        else:
            f1 = 0.0
        return avg_val_loss, accuracy, f1
    
    def evaluate(self) -> Dict[str, Tuple[float, float, float]]:
        """
        Evaluate on all validation datasets.
        Returns: dict like {'cube': (loss, acc, f1), "dodec": (loss, acc, f1), ..., "all": (macro_loss, macro_acc, macro_f1)}
        """
        results: Dict[str, Tuple[float, float, float]] = {}

        val_loaders = {
            'cube': self.cube_loader,
            'dodec': self.dodec_loader,
            'letter_r': self.letter_r_loader,
            'pent': self.pent_loader,
            'star': self.star_loader,
            'stairs': self.stairs_loader,
            'hex_prism': self.hex_prism_loader,
        }

        # Pre-set metrics
        losses, accs, f1s = [], [], []
        for name, loader in val_loaders.items():
            val_loss, val_acc, val_f1 = self._evaluate_one_loader(name, loader)
            results[name] = (val_loss, val_acc, val_f1)
            losses.append(val_loss)
            accs.append(val_acc)
            f1s.append(val_f1)

        # Macro-average across all validation sets
        if losses:
            results['all'] = (sum(losses) / len(losses), sum(accs) / len(accs), sum(f1s) / len(f1s))
        else:
            results['all'] = (0.0, 0.0, 0.0)

        return results
    
    def train(self) -> None:
        """
        Main training loop. Handles optional resume logic and saves checkpoints.
        """
        overall_step = 0
        starting_epoch = 0
        resume_step = None

        # Handle resume logic
        if self.resume_training:
            # Get a list of all directories in the checkpoint directory
            dirs = [os.path.join(self.checkpoint_dir, f.name) for f in os.scandir(self.checkpoint_dir) if f.is_dir()]
            dirs.sort(key=os.path.getctime)
            path = dirs[-1]  # Sorts folders by date modified, most recent checkpoint is the last

            # Extract `epoch_{i}` or `step_{i}`
            basename = os.path.splitext(path)[0]

            self.accelerator.print(f"Resuming from checkpoint: {basename}")
            self.accelerator.load_state(basename)

            if "epoch" in basename:
                starting_epoch = int(basename.replace(os.path.join(self.checkpoint_dir, "epoch_"), "")) + 1
            elif "step" in basename:
                resume_step = int(basename.replace(os.path.join(self.checkpoint_dir, "step_"), ""))
                starting_epoch = resume_step // len(self.train_loader)
                resume_step -= starting_epoch * len(self.train_loader)

        # Training epochs
        for epoch in range(starting_epoch, self.num_epochs):
            overall_step, train_loss = self.train_one_epoch(
                epoch=epoch,
                overall_step=overall_step,
                resume_step=resume_step if (self.resume_training and epoch == starting_epoch) else None
            )
            # Only apply resume_step once
            resume_step = None

            # Validation
            val_results = self.evaluate()

            # Print
            msg_parts = [f"[Epoch {epoch}/{self.num_epochs}] Train Loss: {train_loss:.4f}"]
            for name, (val_loss, val_acc, val_f1) in val_results.items():
                msg_parts.append(
                    f"{name.capitalize()} Val Loss: {val_loss:.4f} | {name.capitalize()} Val Acc: {val_acc:.4f}"
                    f" | {name.capitalize()} Val F1: {val_f1:.4f}"
                )
            self.accelerator.print(" | ".join(msg_parts))

            # Log to Weights & Biases
            if self.log_wandb:
                log_dict = {"Train Loss": train_loss}
                for name, (val_loss, val_acc, val_f1) in val_results.items():
                    log_dict[f"Val Loss/{name}"] = val_loss
                    log_dict[f"Val Acc/{name}"] = val_acc
                    log_dict[f"Val F1/{name}"] = val_f1
                self.accelerator.log(log_dict, step=epoch)

            # Append metrics to <checkpoint_dir>/metrics.jsonl (read by scripts/aggregate_splits.py)
            if self.checkpoint_dir and self.accelerator.is_main_process:
                with open(os.path.join(self.checkpoint_dir, "metrics.jsonl"), "a") as f:
                    f.write(json.dumps({
                        "epoch": epoch,
                        "train_loss": float(train_loss),
                        "val": {
                            name: {"loss": float(l), "acc": float(a), "f1": float(f1)}
                            for name, (l, a, f1) in val_results.items()
                        },
                    }) + "\n")

            # Save checkpoint
            if self.checkpoint_dir:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"epoch_{epoch}")
                self.accelerator.save_state(checkpoint_path)

        self.accelerator.end_training()

def main():
    """
    Main entry point: parse arguments, load configuration,
    initialize the trainer, and start training.
    """
    parser = argparse.ArgumentParser(description='Train ContactTransformer for Contact Prediction')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to the configuration file'
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Create Datasets
    train_dataset = ContactDataset(
        data_dir=config['train_data_dir'],
        horizon=config['horizon'],
    )
    cube_val = ContactDataset(
        data_dir=config['cube_val'],
        horizon=config['horizon'],
    )
    dodec_val = ContactDataset(
        data_dir=config['dodec_val'],
        horizon=config['horizon'],
    )
    letter_r_val = ContactDataset(
        data_dir=config['letter_r_val'],
        horizon=config['horizon'],
    )
    pent_val = ContactDataset(
        data_dir=config['pent_val'],
        horizon=config['horizon'],
    )
    star_val = ContactDataset(
        data_dir=config['star_val'],
        horizon=config['horizon'],
    )
    stairs_val = ContactDataset(
        data_dir=config['stairs_val'],
        horizon=config['horizon'],
    )
    hex_prism_val = ContactDataset(
        data_dir=config['hex_prism_val'],
        horizon=config['horizon'],
    )

    val_dataset = {
        'cube': cube_val,
        'dodec': dodec_val,
        'letter_r': letter_r_val,
        'pent': pent_val,
        'star': star_val,
        'stairs': stairs_val,
        'hex_prism': hex_prism_val,
    }

    # Initialize model
    # contact_model = ContactTransformer(
    #     num_heads=1,
    #     use_vision_attn=True,
    #     asym_ckpt=config['asym_ckpt'],
    # )
    contact_model = CT_without_segmask(
        num_heads=1,
        use_vision_attn=True,
        asym_ckpt=config['asym_ckpt'],
        output_dim=4,
    )

    # Initialize the trainer
    trainer = ContactTrainer(
        config=config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=contact_model,
        checkpoint_dir=config['checkpoint_dir'],
        log_wandb=config['use_wandb'],
        resume_training=False,
        use_stateful_dataloader=False,  # Requires a higher version of PyTorch DataLoader or torchdata
    )

    # Start training
    trainer.train()

if __name__ == '__main__':
    main()