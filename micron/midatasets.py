from functools import reduce
from io import StringIO
import logging
import os

import fsspec
from fsspec.callbacks import TqdmCallback
import gzip

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import ray

from datasets import Dataset, DatasetDict, load_dataset, splits
import tokenizers
import transformers

from datablocks.datablock import Datablock


class DatasetsMixin:
    """
        Must implement `_datasets_paths() -> tuple(train_filename, test_filename)`
    """
    @classmethod
    def read(cls,
             root,
             storage_options={},
             **unused_api_kwargs, 
             ):
        train_path, test_path = cls._dataset_paths(root)
        train_dataframe = pd.read_parquet(train_path, storage_options=storage_options, engine="pyarrow") 
        train_dataset = Dataset.from_pandas(train_dataframe, split='train')
        test_dataframe = pd.read_parquet(test_path, storage_options=storage_options, engine="pyarrow")
        test_dataset = Dataset.from_pandas(test_dataframe, split='test')
        datasets = DatasetDict({splits.Split.TRAIN: train_dataset, splits.Split.TEST: test_dataset})
        return datasets 

    # TODO: use `root` and `fs` instead of `filelist`
    def valid(self, root, **scope):
        train_filename, test_filename = self._dataset_paths()
        filelist = os.listdir(root)
        valid = (train_filename in filelist) and (test_filename in filelist)
        return valid
    
    # TODO: use `root` and `storage_options`
    def metric(self, root, **scope):
        filelist = os.listdir(root)
        train_filename, test_filename = self._dataset_paths()
        metric = (0, 0)
        if (train_filename in filelist) and (test_filename in filelist):
            train_filepath = os.path.join(root, train_filename)
            test_filepath = os.path.join(root, test_filename)
            train_dataframe = pd.read_parquet(train_filepath, engine="pyarrow") 
            test_dataframe = pd.read_parquet(test_filepath, engine="pyarrow")
            metric = (len(train_dataframe), len(test_dataframe))
        return metric

GRCh38_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers"
GRCh38_FILENAME = "GRCh38_latest_genomic"

GRCh38_DATASET_FILENAME = "GRCh38"
GRCh38_VERSION = "0.0.1"

GRCh38_MAX_SEQS = 0
GRCh38_MIN_SUBSEQ_LEN = 10
GRCh38_MAX_SUBSEQ_LEN = 30
GRCh38_TOKENIZER_MAX_LEN = 50
GRCh38_TRAIN_FRACTION = 0.9

class GRCh38(DatasetsMixin):
    version = GRCh38_VERSION
    def __init__(self,
               *,
               verbose=False,
               rm_tmp=True,
               ray_client=None,):
        self.verbose = verbose
        self.rm_tmp = rm_tmp
        self.ray_client = ray_client

    def build(self,
              root,
              storage_options={},
              *, 
              max_seqs=GRCh38_MAX_SEQS,
              min_subseq_len=GRCh38_MIN_SUBSEQ_LEN,
              max_subseq_len=GRCh38_MAX_SUBSEQ_LEN,
              train_fraction=GRCh38_TRAIN_FRACTION,
             ):
        url=GRCh38_URL
        filename = GRCh38_FILENAME
        _fs = fsspec.filesystem('http')
        remote_fna = url+'/'+f'{filename}.fna.gz'
        local_fna = os.path.join(root, f'{filename}.fna.gz')
        if not os.path.isfile(local_fna):
            if self.verbose:
                print(f"Downloading {remote_fna} to {local_fna}")
            _fs.get(remote_fna, local_fna, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_fna}")
        with gzip.open(local_fna, 'r') as seqfile:
            seqstr = seqfile.read().decode()
            seqf = GRCh38._parse_sequences(seqstr)
        if self.rm_tmp:
            if self.verbose:
                print(f"Removing local copy {local_fna}")
            os.remove(local_fna)

        remote_gff = url+'/'+f'{filename}.gff.gz'
        local_gff = os.path.join(root, f'{filename}.gff.gz')
        if not os.path.isfile(local_gff):
            if self.verbose:
                print(f"Downloading {remote_gff} to {local_gff}")
            _fs.get(remote_gff, local_gff, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_gff}")
        with gzip.open(local_gff, 'r') as gfffile:
            gffstr = gfffile.read().decode()
            annf = GRCh38._parse_annotations(gffstr)
        if self.rm_tmp:
            if self.verbose:
                print(f"Removing local copy {local_gff}")
            os.remove(local_gff)

        subseqs_list = self._ray_extract_rna_subseqs(seqf, 
                                                        annf, 
                                                        max_seqs=max_seqs,
                                                        min_subseq_len=min_subseq_len,
                                                        max_subseq_len=max_subseq_len,
                                                        ray_client=self.ray_client,
                                                        verbose=self.verbose)
        if self.verbose:
            print(f"Extracted subsequences, concatenating ...")
        subseqs = pd.concat(subseqs_list)
        if self.verbose:
            print(f"Done concatenating, normalizing ...")
        subseqf = pd.DataFrame({'sequence': subseqs})
        subseqf['sequence'] = subseqf.sequence.apply(GRCh38._normalize_sequence)
        if self.verbose:
            print(f"Done normalizing.  DataFrame complete.")

        train_path, test_path = self._dataset_paths(root)

        train_dataset = Dataset.from_pandas(subseqf.iloc[:int(train_fraction*len(subseqf))], split='train')
        if self.verbose:
            print(f"Built train dataset from dataframe with train_fraction {train_fraction}")
        test_dataset = Dataset.from_pandas(subseqf.iloc[int(train_fraction*len(subseqf)):], split='test')
        if self.verbose:
            print(f"Built test dataset from dataframe with train_fraction {train_fraction}")
        pq.write_table(train_dataset.data.table, train_path)
        if self.verbose:
            print(f"Wrote train dataset to {train_path}")
        pq.write_table(test_dataset.data.table, test_path)
        if self.verbose:
            print(f"Wrote test dataset to {test_path}")
   
    @classmethod
    def read(self,
             root,
             storage_options={},
             **unused_api_kwargs, 
             ):
        train_path, test_path = cls._dataset_paths(root)
        train_dataframe = pd.read_parquet(train_path, storage_options=storage_options, engine="pyarrow") 
        train_dataset = Dataset.from_pandas(train_dataframe, split='train')
        test_dataframe = pd.read_parquet(test_path, storage_options=storage_options, engine="pyarrow")
        test_dataset = Dataset.from_pandas(test_dataframe, split='test')
        datasets = DatasetDict({splits.Split.TRAIN: train_dataset, splits.Split.TEST: test_dataset})
        return datasets 
    
    @staticmethod
    def _dataset_paths(root=None):
        train_path = os.path.join(root, f"{GRCh38_DATASET_FILENAME}.train.parquet",) if root is not None else f"{GRCh38_DATASET_FILENAME}.train.parquet"
        test_path = os.path.join(root, f"{GRCh38_DATASET_FILENAME}.test.parquet",) if root is not None else f"{GRCh38_DATASET_FILENAME}.test.parquet"
        return train_path, test_path

    @staticmethod
    def _parse_annotations(gffstr):
        gffstrs_ = gffstr.split('\n')
        gffstrs = [s for s in gffstrs_ if not s.startswith('#')]
        
        gffio = StringIO('\n'.join(gffstrs))
        gff = pd.read_csv(gffio, sep='\t', names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
        attr = gff.attributes.str.strip()
        gff['ID'] = attr.apply(lambda _: _[3:].split(';')[0].split(':')[0])
        return gff
    
    @staticmethod
    def _parse_sequences(seqstr):
        lines = seqstr.split('\n')
        f_ = pd.DataFrame({'raw': lines})
        
        f_['key'] = None
        f_['is_key'] = f_.raw.str.startswith('>')
        keymask = f_['is_key']
        f_.loc[keymask, 'ID'] = f_[keymask]['raw'].apply(lambda s: s[1:].split(' ')[0])
        f_['key'] = f_['is_key'].cumsum()
        
        keyf = f_[keymask].set_index('key') 
        seqf = pd.DataFrame({'seq': f_.loc[~keymask].groupby('key').apply(lambda g: ''.join(g.raw))})
        seqf['seqid'] = keyf['ID']
        
        seqf['seqlen'] = seqf.seq.str.len()
        seqf['offset'] = seqf['seqlen'].cumsum().shift(1).fillna(0).astype(int)
        return seqf
    
    @staticmethod
    def _normalize_sequence(s):
        s_ = s.upper().replace('T', 'U')
        return s_

    @staticmethod
    def _ray_row_extract_rna_subseqs(rna_ann_row, 
                                     seq_refs,
                                     min_subseq_len,
                                     max_subseq_len,
                                     verbose):
        row = rna_ann_row
        if row.seqid not in seq_refs:
            return pd.Series([])
        seq_ref = seq_refs[row.seqid]
        seq = ray.get(seq_ref)
        start, end = row.start-1, row.end
        N = (end-start)//min_subseq_len
        lens = pd.Series(np.random.randint(min_subseq_len, max_subseq_len, N))
        ends = (start + lens.cumsum()).apply(lambda _: min(_, end))
        starts = ends.shift(1).fillna(start).astype('int')
        
        limits_ = pd.DataFrame({'start': starts, 'end': ends})
        limits = limits_[limits_.start < limits_.end]
        #print(limits)
        subseqs = limits.apply(lambda _: seq[_.start:_.end], axis=1)
        if verbose:
            print(f"Finished subseqs with seqid: {row.seqid}, start: {start}, end: {end}, got {len(subseqs)} subseqs")
        return subseqs
    
    def _ray_extract_rna_subseqs(self, seqs, ann, *, max_seqs=0, min_subseq_len, max_subseq_len, ray_client, verbose):
        if ray_client is None:
            if self.verbose:
                print(f"Spinning up ray_client")
            _ray_client = ray.init(dashboard_host="0.0.0.0")
        else:
            _ray_client = ray_client
        try:

            seq_refs = {row.seqid: ray.put(row.seq) for i, row in seqs.iterrows() if max_seqs == 0 or i < max_seqs}

            rna_ann_ = ann[ann.type.str.find('RNA') != -1]
            rna_ann = rna_ann_[rna_ann_.seqid.isin(seq_refs.keys())]
            with _ray_client:
                subseq_refs = rna_ann.apply(lambda row, seq_refs, min_subseq_len, max_subseq_len, verbose: \
                                            ray.remote(GRCh38._ray_row_extract_rna_subseqs).remote(row, seq_refs, min_subseq_len, max_subseq_len, verbose), 
                                            args=(seq_refs,min_subseq_len, max_subseq_len, verbose), 
                                            axis=1)
                subseqs = [ray.get(ref) for ref in subseq_refs]
        finally:
            if ray_client is None:
                if self.verbose:
                    print(f"Shutting down ray_client")
                del _ray_client
        return subseqs

#GRCh38Datablock = Datablock.define(GRCh38, module_name=__name__, topics=["rna_seqs"], version=GRCh38_VERSION)


MIRNA_VERSION = "0.0.1"
MIRNA_DATASET_URL = "https://mirbase.org/ftp/CURRENT"
MIRNA_DATASET_FILENAME = f"miRNA"
class MiRNA(DatasetsMixin):
    version = MIRNA_VERSION

    def __init__(self, verbose=False, rm_tmp=True, ):
        self.verbose = verbose
        self.rm_tmp = rm_tmp
    
    def build(self,
              root,
              storage_options={},
              *,
              train_fraction=0.9,):
        fs = fsspec.filesystem('http')

        remote_dat = MIRNA_DATASET_URL + '/' + f'{MIRNA_DATASET_FILENAME}.dat.gz'
        local_dat = os.path.join(root, f'{MIRNA_DATASET_FILENAME}.dat.gz')
        if not os.path.isfile(local_dat):
            if self.verbose:
                print(f"Downloading {remote_dat} to {local_dat}")
            fs.get(remote_dat, local_dat, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_dat}")
        with gzip.open(local_dat, 'r') as datfile:
            datstr = datfile.read().decode()
            frame = self._build_frame(datstr)

        train_path, test_path = self._dataset_paths(root)

        train_dataset = Dataset.from_pandas(frame.iloc[:int(train_fraction*len(frame))], split='train')
        if self.verbose:
            print(f"Built train dataset from dataframe with train_fraction {train_fraction}")
        test_dataset = Dataset.from_pandas(frame.iloc[int(train_fraction*len(frame)):], split='test')
        if self.verbose:
            print(f"Built test dataset from dataframe with train_fraction {train_fraction}")
        pq.write_table(train_dataset.data.table, train_path)
        if self.verbose:
            print(f"Wrote train dataset to {train_path}")
        pq.write_table(test_dataset.data.table, test_path)
        if self.verbose:
            print(f"Wrote test dataset to {test_path}")
    
    
    @staticmethod
    def _dataset_paths(root=None):
        train_path = os.path.join(root, f"{MIRNA_DATASET_FILENAME}.train.parquet",) if root is not None else f"{MIRNA_DATASET_FILENAME}.train.parquet"
        test_path = os.path.join(root, f"{MIRNA_DATASET_FILENAME}.test.parquet",) if root is not None else f"{MIRNA_DATASET_FILENAME}.test.parquet"
        return train_path, test_path

    def _build_frame(self, mdstr):
        recs = MiRNA._parse_records(mdstr)
        f = pd.DataFrame.from_records(recs)
        frame = f.sort_values('ID').reset_index(drop=True)
        if self.verbose:
            print(f"Built dataframe")
        return frame

    @staticmethod     
    def _parse_records(mdstr):
        _mdstrs = mdstr.split('\nID')
        mdstrs = [f"ID{s}" for s in _mdstrs]
        _prerecs = [MiRNA._parse_prerecord(s) for s in mdstrs]
        prerecs = [pr for pr in _prerecs if pr['DE'].find('sapiens') != -1]
        recs = [MiRNA._prerecord_to_record(pr) for pr in prerecs]
        return recs

    @staticmethod
    def _parse_prerecord(recstr):
        sqstart = recstr.find('SQ')+2
        sqend = recstr.find('//')
        sq = recstr[sqstart:sqend]
        recstrs = recstr.split('\n')
        rec_ = {s[:2]: s[3:] for s in recstrs}
        _rec = {k: v.strip() for k, v in rec_.items() if k in ['ID', 'AC', 'DE']}
        _rec['SQ'] = sq
        return _rec

    @staticmethod
    def _prerecord_to_record(prerec):
        rec = {}
        _id = prerec['ID'].split(' ')[0]
        rec['ID'] = _id
        _ac = prerec['AC']
        rec['Accession'] = _ac[:-1] if _ac[-1] == ';' else _ac
        sq_ = prerec['SQ']
        sq_strs_ = sq_.split('\n')[1:-1]
        _sq = ''.join([s[:-2].strip() for s in sq_strs_])
        sq = ''.join([s.strip() for s in _sq.split(' ')])
        rec['sequence'] = ''.join([c for c in sq.upper() if c in ['A', 'C', 'G', 'U']])
        return rec

#MiRNADatablock = Datablock.define(MiRNA, module_name=__name__, topics=["rna_seqs"], version=MIRNA_VERSION, use_local_storage=True)


TOKENIZER_VERSION = '0.0.1'
TOKENIZER_TRAINER_CHUNK_SIZE = 200
TOKENIZER_TRAINER_VOCAB_SIZE = 100
TOKENIZER_FILENAME = 'tokenizer'
TOKENIZER_MAX_LEN = 50

class Tokenizer:
    version = TOKENIZER_VERSION

    def __init__(self, verbose=False):
        self.verbose = verbose
    
    def build(self,
              root,
              storage_options,
              *,
              datasets,
              tokenizer_trainer_chunk_size=TOKENIZER_TRAINER_CHUNK_SIZE,
              tokenizer_trainer_vocab_size=TOKENIZER_TRAINER_CHUNK_SIZE,
              ):

        _datasets = datasets
        _tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        #_tokenizer.normalizer = tokenizers.normalizers.Sequence(
        #    []
        #)
        #_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        #    [tokenizers.pre_tokenizers.WhitespaceSplit(), tokenizers.pre_tokenizers.Punctuation()]
        #)
        #_tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        #    [tokenizers.pre_tokenizers.Punctuation()]
        #)
        _tokenizer_trainer = tokenizers.trainers.BpeTrainer(vocab_size=tokenizer_trainer_vocab_size, special_tokens=["<|startoftext|>", "<|endoftext|>"])
        _tokenizer_generator = self._tokenizer_training_corpus_generator(_datasets, chunk_size=tokenizer_trainer_chunk_size) 
        _tokenizer.train_from_iterator(_tokenizer_generator,
                                       trainer=_tokenizer_trainer)
        #_tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
        _tokenizer.decoder = tokenizers.decoders.ByteLevel()

        filepath = os.path.join(root, TOKENIZER_FILENAME)
        _tokenizer.save(filepath)
        if self.verbose:
            print(f"Saved tokenizer to {filepath}")

    def valid(self, root, **scope):
        filename = TOKENIZER_FILENAME
        filelist = os.listdir(root)
        valid = filename in filelist
        return valid
    
    @staticmethod    
    def _tokenizer_training_corpus_generator(dataset_dict, *, chunk_size):
        dd = dataset_dict
        sequence_list = reduce(lambda sequence, ds: sequence+ds['sequence'] if sequence is not None else ds['sequence'], dd.values(), None)
        for i in range(0, len(sequence_list), chunk_size):
            chunk = sequence_list[i:i+chunk_size]
            yield chunk

    def read(self,
             root,
             storage_options={},
             **unused_api_kwargs, 
             ):
        filepath = os.path.join(root, 'tokenizer')
        if self.verbose: 
            print(f"Loading tokenizer from {filepath}")
        _tokenizer = tokenizers.Tokenizer.from_file(filepath)

        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=_tokenizer,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


#TokenizerDatablock = Datablock.define(Tokenizer, topics=["tokenizer"], version=TOKENIZER_VERSION, use_local_storage=True)


TOKENIZED_DATASET_VERSION = '0.0.1'
TOKENIZED_DATASET_MAX_LEN = 10
TOKENIZED_DATASET_NUM_PROC = 24
class TokenizedDatasets(DatasetsMixin):
    def __init__(self, verbose=False, num_proc=TOKENIZED_DATASET_NUM_PROC):
        self.verbose = verbose
        self.num_proc = num_proc
    
    def build(self,
              root,
              storage_options,
              *,
              datasets,
              tokenizer,
              max_len=TOKENIZED_DATASET_MAX_LEN,
              ):
        tokenized_datasets = self.tokenize_datasets(tokenizer, 
                                                    datasets, 
                                                    max_len=max_len,
                                                    num_proc=self.num_proc)
        tokenized_train_path = os.path.join(root, f"tokenized.train.parquet",)
        tokenized_test_path = os.path.join(root, f"tokenized.test.parquet",)
        if self.verbose:
            print(f"Tokenized train and test datasets from {datasets} using tokenizer {tokenizer}")
        pq.write_table(tokenized_datasets['train'].data.table, tokenized_train_path)
        if self.verbose:
            print(f"Wrote tokenized train dataset to {tokenized_train_path}")
        pq.write_table(tokenized_datasets['test'].data.table, tokenized_test_path)
        if self.verbose:
            print(f"Wrote tokenized test dataset to {tokenized_test_path}")

    @staticmethod
    def _dataset_paths(root=None):
        tokenized_train_path = os.path.join(root, f"tokenized.train.parquet",) if root is not None else "tokenized.train.parquet"
        tokenized_test_path = os.path.join(root, f"tokenized.test.parquet",) if root is not None else "tokenized.test.parquet"
        return tokenized_train_path, tokenized_test_path

    def tokenize_datasets(self, tokenizer, datasets, *, max_len, num_proc=1):
        if self.verbose:
            print(f"Tokenizing datasets using {num_proc} procs")
        def tokenize_row(row, *, context_length=max_len):
            outputs = tokenizer(
                row['sequence'],
                truncation=True,
                max_length=max_len,
                return_overflowing_tokens=True,
                return_length=True,
            )
            return outputs
        
        tokenized_datasets = datasets.map(tokenize_row, 
                                          batched=True, 
                                          num_proc=num_proc,
                                          remove_columns=datasets["train"].column_names)
        return tokenized_datasets
