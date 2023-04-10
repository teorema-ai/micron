import os
import pickle

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from datasets import DatasetDict

import plotly.graph_objects as go

import micron.midatasets


def setup_torch(gpu, *, verbose=True):
    if gpu is None:
         raise ValueError(f"GPU is None")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu}"  # This shrinks the GPU universe and maps cuda:0 to {GPU}
    import torch
    print(f"CUDA: device count: {torch.cuda.device_count()}")
    print(f"CUDA: using device(s): {gpu}")
    print(f"CUDA: current (relative) device: {torch.cuda.current_device()}") # This really is device {GPU}

def torch_setup():
     return "WANDB_DISABLED" in os.environ and "CUDA_VISIBLE_DEVICES" in os.environ


GPT2_VERSION = "0.0.1"
GPT2_CONTEXT_LEN = micron.midatasets.TOKENIZER_MAX_LEN
GPT2_NUM_EPOCHS = 3
GPT2_LEARNING_RATE = 2e-5
GPT2_WEIGHT_DECAY = 0.01
GPT2_TRAIN_BATCH_SIZE = 16
GPT2_EVAL_BATCH_SIZE = 16

class GPT2:
    version = GPT2_VERSION
    topics = ['model', 'stats']

    def __init__(self,
                *,
                verbose=False,
                gpu=None,
                ):
        self.verbose = verbose
        self.gpu = gpu
        
    def build(self,
            roots,
            storage_options,
            *,
            tokenized_datasets,
            tokenizer,
            train_max_samples=None,
            test_max_samples=None,
            context_len=GPT2_CONTEXT_LEN,
            num_epochs=GPT2_NUM_EPOCHS,
            learning_rate=GPT2_LEARNING_RATE,
            weight_decay=GPT2_WEIGHT_DECAY,
            train_batch_size=GPT2_TRAIN_BATCH_SIZE,
            eval_batch_size=GPT2_EVAL_BATCH_SIZE,
            new_model_init_weights=False,
            ):
        if not torch_setup():
                setup_torch(self.gpu, verbose=self.verbose)

        model_root = roots['model']
        stats_root = roots['stats']

        model = self._model(tokenizer, context_len)

        data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=f"{model_root}",
            evaluation_strategy="epoch",
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            push_to_hub=False,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=train_batch_size,
            per_device_eval_batch_size=eval_batch_size,
        )

        if self.verbose:
                print(f"Training using training_args: {training_args}")
        try:
            model = model.from_pretrained(model_root)
            new_model = False
        except:
            new_model = True

        model.to(f"cuda:0")

        if new_model and new_model_init_weights:
                model.init_weights()

        if train_max_samples is not None:
            _tokenized_datasets_train = tokenized_datasets['train'].select(range(train_max_samples))
        else:
            _tokenized_datasets_train = tokenized_datasets['train']
        if test_max_samples is not None:
            _tokenized_datasets_test = tokenized_datasets['test'].select(range(test_max_samples))
        else:
            _tokenized_datasets_test = tokenized_datasets['test']
        
        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=data_collator,
            train_dataset=_tokenized_datasets_train,
            eval_dataset=_tokenized_datasets_test,
        )

        if self.verbose:
            print(f"Training with {len(_tokenized_datasets_train)} training examples " + 
                            f"and {len(_tokenized_datasets_test)} test examples")
        trainer.train()
        train_loss_dicts = [{'train_loss': d['loss'], 'epoch': d['epoch']}
            for d in trainer.state.log_history if 'loss' in d]
        eval_loss_dicts = [{'eval_loss': d['eval_loss'], 'epoch': d['epoch']} 
            for d in trainer.state.log_history if 'eval_loss' in d]

        model.to("cpu").save_pretrained(model_root, from_pt=True)

        with open(os.path.join(stats_root, "train_loss_dicts.pickle"), 'wb') as train_loss_file:
            pickle.dump(train_loss_dicts, train_loss_file)
        with open(os.path.join(stats_root, "eval_loss_dicts.pickle"), 'wb') as eval_loss_file:
            pickle.dump(eval_loss_dicts, eval_loss_file)
        

    def read(self, 
             root, 
             storage_options,
             topic,
             *, 
             tokenized_datasets,
             tokenizer,
             train_max_samples=None,
             test_max_samples=None,
             context_len=GPT2_CONTEXT_LEN,
             num_epochs=GPT2_NUM_EPOCHS,
             learning_rate=GPT2_LEARNING_RATE,
             weight_decay=GPT2_WEIGHT_DECAY,
             train_batch_size=GPT2_TRAIN_BATCH_SIZE,
             eval_batch_size=GPT2_EVAL_BATCH_SIZE,
             new_model_init_weights=False,
            ):
        if topic == 'model':
            model = self._model(tokenizer, context_len)
            model.from_pretrained(root)
            return model
        elif topic == 'stats':
            with open(os.path.join(root, "train_loss_dicts.pickle"), 'rb') as train_loss_file:
                train_loss_dicts = pickle.load(train_loss_file)
            with open(os.path.join(root, "eval_loss_dicts.pickle"), 'rb') as eval_loss_file:
                eval_loss_dicts = pickle.load(eval_loss_file)
            return train_loss_dicts, eval_loss_dicts
        else:
            raise ValueError()

    def valid(self, root, topic, **scope):
        if topic not in ['model', 'stats']:
             raise ValueError(f"Unknown topic: {topics}")
        if topic == 'stats':
             _ = (os.path.isfile(os.path.join(root, "train_loss_dicts.pickle"))) and \
                 (os.path.isfile(os.path.join(root, "eval_loss_dicts.pickle")))
        else:
            _ = (os.path.isfile(os.path.join(root, "pytorch_model.bin"))) and \
                (os.path.isfile(os.path.join(root, "config.json"))) and\
                (os.path.isfile(os.path.join(root, "generation_config.json")))
        return _

    @staticmethod
    def plot_losses(train_loss_dicts, eval_loss_dicts, *, show=False):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[d['epoch'] for d in train_loss_dicts], y=[d['train_loss'] for d in train_loss_dicts], name='train_loss'))
        fig.add_trace(go.Scatter(x=[d['epoch'] for d in eval_loss_dicts], y=[d['eval_loss'] for d in eval_loss_dicts], name='eval_loss'))
        if show:
            fig.show()
            train_loss = [d['train_loss'] for d in train_loss_dicts]
            eval_loss = [d['eval_loss'] for d in eval_loss_dicts]
            print(f"train_loss: min: {min(train_loss)}, max: {max(train_loss)}")
            print(f"eval_loss: min: {min(eval_loss)}, max: {max(eval_loss)}")
        return fig

    def _model(self, tokenizer, context_len):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(tokenizer),
            n_ctx=context_len,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config)
        return model
