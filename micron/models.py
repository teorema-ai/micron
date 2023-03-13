import os
import pickle

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from datasets import DatasetDict

import plotly.graph_objects as go

import micron.datasets
from micron import MICRON_CACHE


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


class ModelManager:
    def __init__(self,
               name,
               *,
               cache=MICRON_CACHE,
               version,
               tokenizer_dataset_name,
               tokenized_dataset_name=None,
               train_max_samples=None,
               test_max_samples=None,
               context_len,
               num_epochs=3,
               new_model_init_weights=False,
               learning_rate=2e-5,
               weight_decay=0.01,
               train_batch_size=16,
               eval_batch_size=16,
               gpu=None,
               verbose=False
               ):
         self.name = name
         self.cache = cache
         self.version = version
         self.tokenizer_dataset_name = tokenizer_dataset_name
         self.tokenized_dataset_name = tokenized_dataset_name
         self.train_max_samples = int(train_max_samples) if ((train_max_samples is not None) and (train_max_samples.lower() != 'none')) else None
         self.test_max_samples = int(test_max_samples) if ((test_max_samples is not None) and (test_max_samples.lower() != 'none')) else None
         self.context_len = int(context_len)
         self.num_epochs = int(num_epochs)
         self.new_model_init_weights = bool(new_model_init_weights)
         self.learning_rate = float(learning_rate)
         self.weight_decay = float(weight_decay)
         self.train_batch_size = int(train_batch_size)
         self.eval_batch_size = int(eval_batch_size)
         self.gpu = gpu
         self.verbose = bool(verbose)

    @property
    def tokenized_datasets(self):
        if not hasattr(self, '_tokenized_datasets'):
            tokenized_dataset_manager = \
                micron.datasets.DatasetManager.manager(self.tokenized_dataset_name, 
                                                       tokenized=True,
                                                       cache=self.cache, 
                                                       verbose=self.verbose)
            self._tokenized_datasets = tokenized_dataset_manager.datasets()
            print(f"Loaded tokenized datasets: train: {len(self._tokenized_datasets['train'])}, " + 
                  f"test: {len(self._tokenized_datasets['test'])}")
        return self._tokenized_datasets
        
    @property
    def tokenizer(self):
        if not hasattr(self, '_tokenizer'):
            tokenizer_dataset_manager = micron.datasets.DatasetManager.manager(self.tokenizer_dataset_name, cache=self.cache, verbose=self.verbose)
            self._tokenizer = tokenizer_dataset_manager.tokenizer()
        return self._tokenizer

    def train(self):
        if not torch_setup():
                 setup_torch(self.gpu, verbose=self.verbose)

        model = self._model()

        model_root = os.path.join(self.cache, 
                                  "model", 
                                  self.name, 
                                  self.tokenized_dataset_name, 
                                  f"version={self.version}")

        if self.verbose:
             print(f"Using model cache {model_root}")

        data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        training_args = TrainingArguments(
            output_dir=f"{model_root}",
            evaluation_strategy="epoch",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            push_to_hub=False,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
        )
        if self.verbose:
             print(f"Training using training_args: {training_args}")
        try:
            model = model.from_pretrained(model_root)
            new_model = False
        except:
            new_model = True
        model.to(f"cuda:0")
        if new_model and self.new_model_init_weights:
                model.init_weights()

        if self.tokenized_dataset_name is not None and self.tokenizer_dataset_name:
            if self.train_max_samples is not None:
                 _tokenized_datasets_train = self.tokenized_datasets['train'].select(range(self.train_max_samples))
            else:
                 _tokenized_datasets_train = self.tokenized_datasets['train']
            if self.test_max_samples is not None:
                 _tokenized_datasets_test = self.tokenized_datasets['test'].select(range(self.test_max_samples))
            else:
                 _tokenized_datasets_test = self.tokenized_datasets['test']
            
            trainer = Trainer(
                model=model,
                tokenizer=self.tokenizer,
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
            with open(os.path.join(model_root, "train_loss_dicts.pickle"), 'wb') as train_loss_file:
                pickle.dump(train_loss_dicts, train_loss_file)
            with open(os.path.join(model_root, "eval_loss_dicts.pickle"), 'wb') as eval_loss_file:
                pickle.dump(eval_loss_dicts, eval_loss_file)
            model.to("cuda:0")
        else:
            with open(os.path.join(model_root, "train_loss_dicts.pickle"), 'rb') as train_loss_file:
                train_loss_dicts = pickle.load(train_loss_file)
            with open(os.path.join(model_root, "eval_loss_dicts.pickle"), 'rb') as eval_loss_file:
                eval_loss_dicts = pickle.dump(eval_loss_file)
        return model, train_loss_dict, eval_loss_dict

    @staticmethod
    def plot_losses(train_loss_dicts, eval_loss_dicts, *, show=True):
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
    
GPT2_VERSION = "0"
GPT2_TOKENIZER_DATASET_NAME = "MiRNA"
GPT2_TOKENIZED_DATASET_NAME = "GRCh38"
GPT2_CONTEXT_LEN = micron.datasets.MIRNA_TOKENIZER_MAX_LEN
GPT2_NUM_EPOCHS = 3
GPT2_LEARNING_RATE = 2e-5
GPT2_WEIGHT_DECAY = 0.01
GPT2_TRAIN_BATCH_SIZE = 16
GPT2_EVAL_BATCH_SIZE = 16

class GPT2(ModelManager):
      def __init__(self,
               *,
               cache=MICRON_CACHE,
               version=GPT2_VERSION,
               tokenizer_dataset_name=GPT2_TOKENIZER_DATASET_NAME,
               tokenized_dataset_name=GPT2_TOKENIZED_DATASET_NAME,
               train_max_samples=None,
               test_max_samples=None,
               context_len=GPT2_CONTEXT_LEN,
               num_epochs=GPT2_NUM_EPOCHS,
               learning_rate=GPT2_LEARNING_RATE,
               weight_decay=GPT2_WEIGHT_DECAY,
               train_batch_size=GPT2_TRAIN_BATCH_SIZE,
               eval_batch_size=GPT2_EVAL_BATCH_SIZE,
               new_model_init_weights=False,
               gpu=None,
               verbose=False
               ):
           super().__init__(self.__class__.__name__,
                            cache=cache,
                            version=version,
                            tokenized_dataset_name=tokenized_dataset_name,
                            tokenizer_dataset_name=tokenizer_dataset_name,
                            train_max_samples=train_max_samples,
                            test_max_samples=test_max_samples,
                            context_len=context_len,
                            num_epochs=num_epochs,
                            new_model_init_weights=new_model_init_weights,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            train_batch_size=train_batch_size,
                            eval_batch_size=eval_batch_size,
                            gpu=gpu,
                            verbose=verbose)
      def _model(self):
        config = AutoConfig.from_pretrained(
            "gpt2",
            vocab_size=len(self.tokenizer),
            n_ctx=self.context_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        model = GPT2LMHeadModel(config)
        return model
