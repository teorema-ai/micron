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


from micron import MICRON_CACHE

TOKENIZER_TRAINER_CHUNK_SIZE = 200
TOKENIZER_TRAINER_VOCAB_SIZE = 100

class DatasetManager:
    def __init__(
            self,
            name,
            *, 
            cache=MICRON_CACHE,
            train_fraction=0.9,
            tokenizer_trainer_chunk_size=TOKENIZER_TRAINER_CHUNK_SIZE,
            tokenizer_trainer_vocab_size=TOKENIZER_TRAINER_VOCAB_SIZE,
            tokenized=False,
            tokenizer_dataset_name=None,
            tokenizer_num_proc=24,
            tokenizer_max_len=10,
            rebuild=False,
            manage_ray=True,
            verbose=False):
        self.name = name
        self.cache = cache
        self.train_fraction = float(train_fraction)
        self.tokenized = bool(tokenized)
        self.tokenizer_dataset_name = tokenizer_dataset_name
        if self.tokenizer_dataset_name is None or self.tokenizer_dataset_name.lower() == 'none':
            self.tokenizer_dataset_name = self.__class__.__name__
        self.tokenizer_max_len = int(tokenizer_max_len)
        self.tokenizer_num_proc = int(tokenizer_num_proc)
        self.tokenizer_trainer_chunk_size = int(tokenizer_trainer_chunk_size)
        self.tokenizer_trainer_vocab_size = int(tokenizer_trainer_vocab_size)
        self.verbose = bool(verbose)
        self.rebuild = bool(rebuild)
        self.manage_ray = bool(manage_ray)
        self.root = os.path.join(self.cache, 'datasets', self.name)
        
    @staticmethod
    def manager(name, **kwargs):
        return globals()[name](**kwargs)
    
    def datasets(self):
        if self.verbose:
            print(f"Using cache: {self.cache}")
        
        os.makedirs(self.root, exist_ok=True)

        if self.tokenized:
            tokenized_train_path = os.path.join(self.root, f"{self.name}_train_tokenized.parquet")
            tokenized_test_path = os.path.join(self.root, f"{self.name}_test_tokenized.parquet")
            if not self.rebuild:
                if os.path.isfile(tokenized_train_path) and os.path.isfile(tokenized_test_path):
                    if self.verbose: 
                        print(f"Loading tokenized datasets from {tokenized_train_path} and {tokenized_test_path}")
                        _tokenized_datasets = load_dataset("parquet", 
                                    data_files={splits.Split.TRAIN: tokenized_train_path, splits.Split.TEST: tokenized_test_path},)
                        return _tokenized_datasets

        train_path = os.path.join(self.root, f"{self.name}_train.parquet")
        test_path = os.path.join(self.root, f"{self.name}_test.parquet")

        if not os.path.isfile(train_path) or not os.path.isfile(test_path) or self.rebuild:
            frame = self._build_dataframe()
            train_dataset = Dataset.from_pandas(frame.iloc[:int(self.train_fraction*len(frame))], split='train')
            print(f"Built train dataset from dataframe for {self.name}")
            test_dataset = Dataset.from_pandas(frame.iloc[int(self.train_fraction*len(frame)):], split='test')
            print(f"Built test dataset from dataframe for {self.name}")
            pq.write_table(train_dataset.data.table, train_path)
            print(f"Wrote train dataset for {self.name}")
            pq.write_table(test_dataset.data.table, test_path)
            print(f"Wrote test dataset for {self.name}")
        else:
            if self.verbose:
                print(f"Using caches: {train_path} and {test_path}")
        datasets = load_dataset("parquet", 
                                    data_files={splits.Split.TRAIN: train_path, splits.Split.TEST: test_path},)
        if self.tokenized:
            tokenizer_mgr = self.manager(self.tokenizer_dataset_name, cache=self.cache, verbose=self.verbose)
            tokenizer = tokenizer_mgr.tokenizer()
            if tokenizer is None:
                raise ValueError(f"Failed to obtain tokenizer {self.tokenizer_dataset_name}")
            if self.verbose:
                print(f"Tokenizing datasets using column 'sequence' and tokenizer_max_len: {self.tokenizer_max_len}")
            tokenized_datasets = self.tokenize_datasets(tokenizer, 
                                                              datasets, 
                                                              max_len=self.tokenizer_max_len,
                                                              num_proc=self.tokenizer_num_proc)
            print(f"Tokenized train and test datasets for {self.name}")
            pq.write_table(tokenized_datasets['train'].data.table, tokenized_train_path)
            print(f"Wrote tokenized train dataset for {self.name}")
            pq.write_table(tokenized_datasets['test'].data.table, tokenized_test_path)
            print(f"Wrote tokenized test dataset for {self.name}")
            return tokenized_datasets
        else:
            return datasets
    
    def tokenizer(self):
        path = os.path.join(self.root, f"{self.name}_tokenizer.json")
        if not os.path.isfile(path) or self.rebuild:
            _datasets = self.manager(self.name, cache=self.cache, verbose=self.verbose).datasets()
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
            _tokenizer_trainer = tokenizers.trainers.BpeTrainer(vocab_size=self.tokenizer_trainer_vocab_size, special_tokens=["<|startoftext|>", "<|endoftext|>"])
            _tokenizer.train_from_iterator(self._tokenizer_training_corpus_generator(_datasets, chunk_size=self.tokenizer_trainer_chunk_size), 
                                           trainer=_tokenizer_trainer)
            #_tokenizer.post_processor = tokenizers.processors.ByteLevel(trim_offsets=False)
            _tokenizer.decoder = tokenizers.decoders.ByteLevel()
            if self.verbose:
                print(f"Saving tokenizer {self.name} in {path}")
            _tokenizer.save(path)
        
        if self.verbose: 
            print(f"Loading tokenizer {self.name} from {path}")
        _tokenizer = tokenizers.Tokenizer.from_file(path)

        tokenizer = transformers.PreTrainedTokenizerFast(
            tokenizer_object=_tokenizer,
            bos_token="<|startoftext|>",
            eos_token="<|endoftext|>",
        )
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
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
    
    @staticmethod    
    def _tokenizer_training_corpus_generator(dataset_dict, *, chunk_size):
        dd = dataset_dict
        sequence_list = reduce(lambda sequence, ds: sequence+ds['sequence'], dd.values(), [])
        for i in range(0, len(sequence_list), chunk_size):
            yield sequence_list[i:i+chunk_size]


 
GRCh38_DATASET_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers"
GRCh38_DATASET_FILENAME = f"GRCh38_latest_genomic"
GRCh38_MIN_SUBSEQ_LEN = 10
GRCh38_MAX_SUBSEQ_LEN = 30
GRCh38_TOKENIZER_MAX_LEN = 50
class GRCh38(DatasetManager):
    def __init__(self,
              *, 
              cache=MICRON_CACHE,
              train_fraction=0.9,
              tokenized=False,
              tokenizer_dataset_name=None,
              tokenizer_num_proc=24,
              tokenizer_max_len=GRCh38_TOKENIZER_MAX_LEN,
              rebuild=False,
              verbose=False,
              url=GRCh38_DATASET_URL,
              max_seqs=None,
              min_subseq_len=GRCh38_MIN_SUBSEQ_LEN,
              max_subseq_len=GRCh38_MAX_SUBSEQ_LEN,
              manage_ray=True,):
        super().__init__(self.__class__.__name__, 
                         cache=cache, 
                         train_fraction=train_fraction,
                         tokenized=tokenized,
                         tokenizer_dataset_name=tokenizer_dataset_name,
                         tokenizer_num_proc=tokenizer_num_proc,
                         tokenizer_max_len=tokenizer_max_len,
                         rebuild=rebuild,
                         verbose=verbose)
        self.url = url
        self.max_seqs = int(max_seqs) if max_seqs is not None else None
        self.min_subseq_len = int(min_subseq_len)
        self.max_subseq_len = int(max_subseq_len)
        self.manage_ray = bool(manage_ray)
        
    def _build_dataframe(self):
        fs = fsspec.filesystem('http')
        remote_fna = self.url+'/'+f'{GRCh38_DATASET_FILENAME}.fna.gz'
        local_fna = os.path.join(self.root, f'{GRCh38_DATASET_FILENAME}.fna.gz')
        if not os.path.isfile(local_fna) or self.rebuild:
            if self.verbose:
                print(f"Downloading {remote_fna} to {local_fna}")
            fs.get(remote_fna, local_fna, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_fna}")
        with gzip.open(local_fna, 'r') as seqfile:
            seqstr = seqfile.read().decode()
            seqf = GRCh38._parse_sequences(seqstr)

        remote_gff = self.url+'/'+f'{GRCh38_DATASET_FILENAME}.gff.gz'
        local_gff = os.path.join(self.root, f'{GRCh38_DATASET_FILENAME}.gff.gz')
        if not os.path.isfile(local_gff) or self.rebuild:
            if self.verbose:
                print(f"Downloading {remote_gff} to {local_gff}")
            fs.get(remote_gff, local_gff, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_gff}")
        with gzip.open(local_gff, 'r') as gfffile:
            gffstr = gfffile.read().decode()
            annf = GRCh38._parse_annotations(gffstr)

        subseqs_list = GRCh38._ray_extract_rna_subseqs(seqf, 
                                                        annf, 
                                                        max_seqs=self.max_seqs,
                                                        min_subseq_len=self.min_subseq_len,
                                                        max_subseq_len=self.max_subseq_len,
                                                        manage_ray=self.manage_ray,
                                                        verbose=self.verbose)
        print(f"Extracted subsequences, concatenating ...")
        subseqs = pd.concat(subseqs_list)
        print(f"Done concatenating, normalizing ...")
        subseqf = pd.DataFrame({'sequence': subseqs})
        subseqf['sequence'] = subseqf.sequence.apply(GRCh38._normalize_sequence)
        print(f"Done normalizing.  DataFrame complete.")
        return subseqf
    
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
    
    @staticmethod
    def _ray_extract_rna_subseqs(seqs, ann, *, max_seqs=None, min_subseq_len, max_subseq_len, manage_ray, verbose):
        if manage_ray:
            try:
                ray.shutdown()
            except:
                pass
            ray.init(dashboard_host="0.0.0.0")

        seq_refs = {row.seqid: ray.put(row.seq) for i, row in seqs.iterrows() if max_seqs is None or i < max_seqs}

        rna_ann_ = ann[ann.type.str.find('RNA') != -1]
        rna_ann = rna_ann_[rna_ann_.seqid.isin(seq_refs.keys())]
        subseq_refs = rna_ann.apply(lambda row, seq_refs, min_subseq_len, max_subseq_len, verbose: \
                                    ray.remote(GRCh38._ray_row_extract_rna_subseqs).remote(row, seq_refs, min_subseq_len, max_subseq_len, verbose), 
                                    args=(seq_refs,min_subseq_len, max_subseq_len, verbose), 
                                    axis=1)
        subseqs = [ray.get(ref) for ref in subseq_refs]

        if manage_ray:
            ray.shutdown()
        return subseqs
    

MIRNA_DATASET_URL = "https://mirbase.org/ftp/CURRENT"
MIRNA_DATASET_FILENAME = f"miRNA"
MIRNA_TOKENIZER_MAX_LEN = 30

class MiRNA(DatasetManager):
    def __init__(self,
              *,
              cache=MICRON_CACHE,
              train_fraction=0.9,
              tokenized=False,
              tokenizer_dataset_name=None,
              tokenizer_num_proc=1,
              tokenizer_max_len=MIRNA_TOKENIZER_MAX_LEN,
              rebuild=False,
              verbose=False,
              url=MIRNA_DATASET_URL,):
        super().__init__(self.__class__.__name__, 
                         cache=cache, 
                         train_fraction=train_fraction,
                         tokenized=tokenized,
                         tokenizer_dataset_name=tokenizer_dataset_name,
                         tokenizer_num_proc=tokenizer_num_proc,
                         tokenizer_max_len=tokenizer_max_len,
                         rebuild=rebuild,
                         verbose=verbose)
        self.url = url

    def _build_dataframe(self):
        fs = fsspec.filesystem('http')

        remote_dat = self.url+'/'+f'{MIRNA_DATASET_FILENAME}.dat.gz'
        local_dat = os.path.join(self.root, f'{MIRNA_DATASET_FILENAME}.dat.gz')
        if not os.path.isfile(local_dat) or self.rebuild:
            if self.verbose:
                print(f"Downloading {remote_dat} to {local_dat}")
            fs.get(remote_dat, local_dat, callback=TqdmCallback())
        if self.verbose:
            print(f"Parsing local copy {local_dat}")
        with gzip.open(local_dat, 'r') as datfile:
            datstr = datfile.read().decode()
            mf = self._get_frame(datstr)
            return mf
               
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

    @staticmethod     
    def _get_records(mdstr):
        _mdstrs = mdstr.split('\nID')
        mdstrs = [f"ID{s}" for s in _mdstrs]
        _prerecs = [MiRNA._parse_prerecord(s) for s in mdstrs]
        prerecs = [pr for pr in _prerecs if pr['DE'].find('sapiens') != -1]
        recs = [MiRNA._prerecord_to_record(pr) for pr in prerecs]
        return recs

    def _get_frame(self, mdstr):
        recs = MiRNA._get_records(mdstr)
        f = pd.DataFrame.from_records(recs)
        frame = f.sort_values('ID').reset_index(drop=True)
        print(f"Built dataframe for {self.name}")
        return frame
    


