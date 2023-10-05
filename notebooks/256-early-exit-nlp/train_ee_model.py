import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import get_linear_schedule_with_warmup



def preprocess_squad2_dataset(tokenizer=None, num_samples=-1):
    """Function to preprocess and tokenize squad dataset before training model"""
    data_split = "train" if num_samples == -1 else f"train[:{num_samples}]"
    squad = load_dataset("squad_v2", split=data_split)

    # Taken from HuggingFace Question-Answering tutorial
    def preprocess_function(examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = tokenizer(
            questions,
            examples["context"],
            max_length=384,
            truncation="only_second",
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            answer = answers[i]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label it (0, 0)
            if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    return squad.map(preprocess_function, batched=True, remove_columns=squad.column_names)


class EarlyExitModelTrainer:
    def __init__(self, model=None, save_dir="./ee_model_ckpt"):
        self.model = model
        self.save_dir = save_dir
        self.training_phase = 1
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def set_training_phase(self, phase=1):
        if phase == self.training_phase:
            return
        
        if phase == 1:
            ## Freeze n-1 ramp classifiers
            for params in self.model.ramp_classifiers[:-1].parameters():
                params.requires_grad = False
            self.training_phase = 1
        elif phase == 2:
            ## Freeze all parameters execept n-1 ramp classifiers
            for params in self.model.parameters():
                params.requires_grad = False    
            for params in self.model.ramp_classifiers[:-1].parameters():
                params.requires_grad = True
            self.training_phase = 2
        else:
            raise ValueError('Training phase can be 1 or 2 only.')

    def get_training_phase(self):
        return self.training_phase

    def save_model_checkpoint(self):
        self.model.save_pretrained(self.save_dir)

    def train(self, dataset=None, **kwargs):
        # Set up training configuration
        batch_size = kwargs.get("batch_size", 32)
        lr = kwargs.get("lr", 3e-5)
        weight_decay = kwargs.get("weight_decay", 0.01)
        num_epochs = kwargs.get("num_epochs", 3)
        warmup_proportion = kwargs.get("warmup_proportion", 0.2)

        # Define dataloader
        loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=8, pin_memory=True) if self.device == torch.device("cuda") else dict(shuffle=True, batch_size=batch_size)
        train_loader = DataLoader(dataset, **loader_args)

        # Prepare for model training
        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        num_training_steps = len(train_loader)*num_epochs
        num_warmup_steps = int(warmup_proportion*training_steps)  
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

        # Training loop
        for epoch in range(num_epochs):
            avg_step_loss = 0
            num_steps = 0

            for qa_pair in train_loader:
                optimizer.zero_grad()

                input_ids = torch.transpose(torch.stack(qa_pair['input_ids']), 0, 1).to(device)
                attention_mask = torch.transpose(torch.stack(qa_pair['attention_mask']), 0, 1).to(device)
                start_positions = qa_pair['start_positions'].to(device)
                end_positions = qa_pair['end_positions'].to(device)

                output =  model(input_ids=input_ids, attention_mask=attention_mask, start_positions=start_positions, end_positions=end_positions, training=True, training_phase=self.training_phase)
                loss = output.loss
                avg_step_loss += loss

                loss.backward()
                optimizer.step()
                scheduler.step()
                num_steps += 1
  
            avg_step_loss /= num_steps
            print(f"Epoch {epoch+1} Complete, Average Loss: {avg_step_loss}")

            # Save model checkpoint locally after each epoch
            self.save_model_checkpoint()
            print(f"Saved model checkpoint after epoch {epoch+1} in")
