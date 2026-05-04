import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from transformers import CLIPProcessor, CLIPModel, GPT2Tokenizer, GPT2LMHeadModel
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Metric
import wandb


class ImageNetDataset(Dataset):
    def __init__(self, root, transform=None, meta_file=None):
        self.dataset = datasets.ImageFolder(root=root, transform=transform)
        self.wnid_to_labels = self._load_meta(meta_file) if meta_file else None
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_meta(self, meta_file):
        meta_data = torch.load(meta_file)

        meta_dict = meta_data[0] 
        return {wnid: names[0] for wnid, names in meta_dict.items()}


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label_idx = self.dataset[idx]  
        label_wnid = self.dataset.classes[label_idx]  
        label_text = self.wnid_to_labels[label_wnid] if self.wnid_to_labels else label_wnid
        return image, label_text

# MLP
class MLPMapping(nn.Module):
    def __init__(self, input_dim=512, output_dim=768, prefix_length=10, hidden_dim=1024):
        super(MLPMapping, self).__init__()
        self.prefix_length = prefix_length
        self.output_dim = output_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim * prefix_length)
        )

    def forward(self, x):
        out = self.mlp(x)
        return out.view(-1, self.prefix_length, self.output_dim)

# Lightning 
        
        
        
class ImageNetTrainer(pl.LightningModule):
    def __init__(self, clip_model_name="openai/clip-vit-base-patch32", gpt2_model_name="gpt2", lr=1e-4):
        super(ImageNetTrainer, self).__init__()
        self.save_hyperparameters()
        
        # model definition
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_name).eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.mapping_network = MLPMapping(input_dim=512, output_dim=768, prefix_length=10)

        # loss function
        self.ce_loss_fn = nn.CrossEntropyLoss()
        self.cosine_similarity = nn.CosineSimilarity(dim=-1)
        self.lr = lr
        
        self.val_outputs = []

    def forward(self, images):
        
        mean = torch.tensor([0.485, 0.456, 0.406]).to(images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).to(images.device).view(1, 3, 1, 1)
        images = images * std + mean  

        images = images.clamp(0, 1)
        
        # CLIP feature extracting
        inputs = self.clip_processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            clip_features = self.clip_model.get_image_features(**inputs)  # (batch_size, 512)

        # MLP 
        prefix_embeds = self.mapping_network(clip_features)  # (batch_size, prefix_length, 768)
        return prefix_embeds

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        print('\n\n training step begin', batch_idx)
        images, labels = batch
        batch_size = len(labels)
        print("labels from dataset: ",labels)
        images = images.to(self.device)
        labels = [self.tokenizer(label, return_tensors="pt").input_ids.squeeze(0).to(self.device) for label in labels]

        # Step 1: forward 
        prefix_embeds = self(images)
        

        #Step 2: Prompt tokenization
        prompt = "Please generate a label for the image:"
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_embeds = self.gpt2_model.transformer.wte(prompt_ids)  # (1, prompt_length, 768)
        inputs_embeds = torch.cat([prompt_embeds.expand(prefix_embeds.size(0), -1, -1), prefix_embeds], dim=1)
        
        
        
        # Step 3: Compute logits (manual generation)
        max_new_tokens = 2
        num_return_sequences = 3
        total_loss = 0.0
        correct = 0
        batch_generated_labels = []
        
        for batch_idx in range(batch_size):
            # Prepare ground truth
            ground_truth = labels[batch_idx]
            if ground_truth.size(0) < max_new_tokens:
                pad_token = torch.tensor([self.tokenizer.pad_token_id], device=self.device, dtype=torch.long)
                ground_truth = torch.cat([ground_truth, pad_token.repeat(max_new_tokens - ground_truth.size(0))], dim=0)
            elif ground_truth.size(0) > max_new_tokens:
                ground_truth = ground_truth[-max_new_tokens:]
            ground_truth_text = self.tokenizer.decode(ground_truth, skip_special_tokens=True)

            # Generate sequences and calculate loss
            sample_loss = 0.0
            current_inputs = inputs_embeds[batch_idx:batch_idx + 1].clone()  # Shape: (1, seq_len, embed_dim)
            generated_labels = []
            correct_flag = False
            
            for seq_idx in range(num_return_sequences):
                sequence_logits = []
                generated_sequence = []
                for _ in range(max_new_tokens):
                    outputs = self.gpt2_model(inputs_embeds=current_inputs, return_dict=True)
                    logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
                    sequence_logits.append(logits)

                    # Use argmax for deterministic token generation
                    next_token = logits.argmax(dim=-1, keepdim=True)  # Shape: (1, 1)
                    generated_sequence.append(next_token.item())
                    next_token_embeds = self.gpt2_model.transformer.wte(next_token)
                    current_inputs = torch.cat([current_inputs, next_token_embeds], dim=1)

                generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                generated_labels.append(generated_text)
                
                # Stack logits and compute loss
                sequence_logits = torch.cat(sequence_logits, dim=0)  # Shape: (max_new_tokens, vocab_size)
                seq_loss = F.cross_entropy(sequence_logits, ground_truth)
                sample_loss += seq_loss
                
                # Check if generated label matches ground truth
                if generated_text == ground_truth_text:
                    correct_flag = True

            batch_generated_labels.append(generated_labels)
            
            if correct_flag:
                correct += 1
            
            # Average loss across sequences
            sample_loss /= num_return_sequences
            total_loss += sample_loss

        # Average loss across batch
        total_loss /= batch_size
        accuracy = correct / batch_size

        # Log metrics
        print(f"Generated labels for batch {batch_idx}: {batch_generated_labels}")
        self.log("train_loss", total_loss)
        self.log("train_accuracy", accuracy)
        return total_loss        
        

    def validation_step(self, batch, batch_idx):
        print('\n\n validation step begin', batch_idx)
        torch.cuda.empty_cache()
        images, labels = batch
        batch_size = len(labels)
        print("labels from dataset: ",labels)
        images = images.to(self.device)
        labels = [self.tokenizer(label, return_tensors="pt").input_ids.squeeze(0).to(self.device) for label in labels]

        # Step 1: forward 
        prefix_embeds = self(images)
        

        #Step 2: Prompt tokenization
        prompt = "Please generate a label for the image:"
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_embeds = self.gpt2_model.transformer.wte(prompt_ids)  # (1, prompt_length, 768)
        inputs_embeds = torch.cat([prompt_embeds.expand(prefix_embeds.size(0), -1, -1), prefix_embeds], dim=1)
        
        
        
        # Step 3: Compute logits (manual generation)
        max_new_tokens = 2
        num_return_sequences = 3
        total_loss = 0.0
        correct = 0
        batch_generated_labels = []
        
        for batch_idx in range(batch_size):
            # Prepare ground truth
            ground_truth = labels[batch_idx]
            if ground_truth.size(0) < max_new_tokens:
                pad_token = torch.tensor([self.tokenizer.pad_token_id], device=self.device, dtype=torch.long)
                ground_truth = torch.cat([ground_truth, pad_token.repeat(max_new_tokens - ground_truth.size(0))], dim=0)
            elif ground_truth.size(0) > max_new_tokens:
                ground_truth = ground_truth[-max_new_tokens:]
            ground_truth_text = self.tokenizer.decode(ground_truth, skip_special_tokens=True)

            # Generate sequences and calculate loss
            sample_loss = 0.0
            current_inputs = inputs_embeds[batch_idx:batch_idx + 1].clone()  # Shape: (1, seq_len, embed_dim)
            generated_labels = []
            correct_flag = False
            
            for seq_idx in range(num_return_sequences):
                sequence_logits = []
                generated_sequence = []
                for _ in range(max_new_tokens):
                    outputs = self.gpt2_model(inputs_embeds=current_inputs, return_dict=True)
                    logits = outputs.logits[:, -1, :]  # Shape: (1, vocab_size)
                    sequence_logits.append(logits)

                    # Use argmax for deterministic token generation
                    next_token = logits.argmax(dim=-1, keepdim=True)  # Shape: (1, 1)
                    generated_sequence.append(next_token.item())
                    next_token_embeds = self.gpt2_model.transformer.wte(next_token)
                    current_inputs = torch.cat([current_inputs, next_token_embeds], dim=1)

                generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
                generated_labels.append(generated_text)
                
                # Stack logits and compute loss
                sequence_logits = torch.cat(sequence_logits, dim=0)  # Shape: (max_new_tokens, vocab_size)
                seq_loss = F.cross_entropy(sequence_logits, ground_truth)
                sample_loss += seq_loss
                
                # Check if generated label matches ground truth
                if generated_text == ground_truth_text:
                    correct_flag = True

            batch_generated_labels.append(generated_labels)
            
            if correct_flag:
                correct += 1
            
            # Average loss across sequences
            sample_loss /= num_return_sequences
            total_loss += sample_loss

        # Average loss across batch
        total_loss /= batch_size
        accuracy = correct / batch_size

        print(f"Generated labels for batch {batch_idx}: {batch_generated_labels}")    
        
        self.val_outputs.append({"val_loss": total_loss, "val_accuracy": torch.tensor(accuracy)})


    def on_validation_epoch_end(self):
        avg_loss = torch.stack([x["val_loss"] for x in self.val_outputs]).mean()
        
        avg_accuracy = torch.stack([x["val_accuracy"] for x in self.val_outputs]).mean()

        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_accuracy)
        
        self.val_outputs.clear()

        # save checkpoint
        checkpoint_path = f"checkpoint3_epoch_{self.current_epoch}.pth"
        torch.save(self.mapping_network.state_dict(), checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
        
        
    def configure_optimizers(self):
        return optim.AdamW(self.mapping_network.parameters(), lr=self.lr)




def load_checkpoint(checkpoint_path, model):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.mapping_network.load_state_dict(checkpoint)
    print(f"Checkpoint loaded from {checkpoint_path}")
    
    
        
        
        
        
        


wandb.login(key = "437d367183fc96c3c8cb60c16e92242884c4a24c")

# dataloader
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_root = "../../../../data/public/imagenet2012/train"  
val_root = "../../../../data/public/imagenet2012/val"  
meta_file = "../../../../data/public/imagenet2012/meta.bin"
    
    
train_set = ImageNetDataset(root=train_root, transform=transform, meta_file=meta_file)
val_set = ImageNetDataset(root=val_root, transform=transform, meta_file=meta_file)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)

# WandB ini
wandb.init(project="image label generation", name= '2.0')

# Lightning Trainer
trainer = pl.Trainer(
    max_epochs=4,
    log_every_n_steps=2,
    val_check_interval=0.1,
    logger=pl.loggers.WandbLogger(project="image label generation")
)

# call training 
model = ImageNetTrainer()
resume_checkpoint_path = "checkpoint3_epoch_1.pth" 

if resume_checkpoint_path:
    load_checkpoint(resume_checkpoint_path, model)
    
trainer.fit(model, train_loader, val_loader)


#ImageNet 1,281,167 / batch 64 = Epoch_step 20000