from dataclasses import dataclass
import functools
import importlib
import inspect
from io import StringIO
from itertools import combinations
import gzip
import os
import pickle
from sklearn.cluster import KMeans
import tarfile
import tempfile
from typing import Optional, Dict, List

import fsspec
from fsspec.callbacks import TqdmCallback

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from Bio import Entrez
from Bio import SeqIO
Entrez.email = "fight.cancer@quantum.bio.tech"

import fasttext
import matplotlib.pyplot as plt
import plotly.express as px
import umap

from micron.cclustering import ZSConsensusClustering


class Datablock:
    REVISION = '0.0.1'
    @dataclass
    class SCOPE:
        pass

    @staticmethod
    def display_umap(frame, *, color=None, seed=42):
        _umap = umap.UMAP(random_state=seed)
        _udata = _umap.fit_transform(frame.fillna(0.0))
        plt.scatter(_udata[:, 0], _udata[:, 1], c=color)

    def __init__(self, 
                 roots=None, 
                 filesystem:fsspec.AbstractFileSystem = fsspec.filesystem("file"), 
                 scope=None,
                 *, 
                 verbose=False, 
                 debug=False, 
                 rm_tmp=True, ):
        self.scope = scope
        if self.scope is None:
            self.scope = self.SCOPE()
        self.roots = roots
        self.filesystem = filesystem
        self.verbose = verbose
        self.debug = debug
        self.rm_tmp = rm_tmp
        if hasattr(self, '__setup__'):
            self.__setup__()

    def valid(self, topic=None):
        if hasattr(self, 'TOPICS'):
            path = self.path(topic)
        else:
            path = self.path()
        if path is None:
            return True # by default
        self.print_debug(f"{self.__class__}: valid(): checking path {path} to validate topic {repr(topic)}")
        _ = self.filesystem.exists(path)
        self.print_debug(f"{self.__class__}: valid(): path {path} exists: {_}")
        return _            

    def built(self):
        built = True
        if hasattr(self, 'TOPICS'):
            for topic in self.TOPICS:
                if not self.valid(topic):
                    self.print_verbose(f"Topic {topic} not built")
                    built = False
                    break
        else:
            if not self.valid():
                built = False
        return built

    def path(self, topic=None):
        roots = self.roots
        filesystem = self.filesystem
        if topic is not None:
            if filesystem.protocol == 'file':
                if roots is None:
                    path = os.path.join(os.getcwd(), self.TOPICS[topic])
                else:
                    path_ = roots[topic]
                    os.makedirs(path_, exist_ok=True)
                    path = os.path.join(path_, self.TOPICS[topic])
            else:
                path = roots[topic] + "/" + self.TOPICS[topic]
        else:
            if filesystem.protocol == 'file':
                if roots is None:
                    path = os.join(os.getcwd(), self.FILENAME)
                else:
                    path_ = roots
                    os.makedirs(path_, exist_ok=True)
                    path = os.path.join(path_, self.FILENAME)
            else:
                path = roots + "/" + self.FILENAME
        return path

    def print_verbose(self, s):
        if self.verbose:
            print(f">>> {self.__class__.__qualname__}: {s}")

    def print_debug(self, s):
        if self.debug:
            print(f"DEBUG: >>> {self.__class__.__qualname__}: {s}")


class miRLogCoHN(Datablock):
    """
        Data for the clustering HNSC study described in from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854517/.
        TODO: do not save 'pivots' or 'downregulated_mirna_infixes' to a file, return them from code instead?
    """
    REVISION = "0.1.1"
    FILENAME = "mircohn_rpm_log2.parquet"

    _SRC_URL = "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/HNSC/20160128/"
    _SRC_TAR_DIRNAME = "gdac.broadinstitute.org_HNSC.miRseq_Mature_Preprocess.Level_3.2016012800.0.0"
    _SRC_DAT_FILENAME = "HNSC.miRseq_mature_RPM_log2.txt"

    def read(self):
        self.print_verbose(f"Reading '{self.__class__.__qualname__}'")
        topic_tgt_path = self.path()
        topic_frame = pd.read_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
        return topic_frame
    
    def build(self):
        """
            Generate a pandas dataframe of TCGA HNSC mature MiRNA sequence samples.
        """
        if self.built():
            self.print_verbose("Already built.  Done.")
        else:
            self.print_verbose("Building ...")
            # logcounts
            topic_tgt_path = self.path()
            fs = fsspec.filesystem('http')
            with tempfile.TemporaryDirectory() as tmpdir:
                remote_tarpath = self._SRC_URL + '/' + self._SRC_TAR_DIRNAME + ".tar.gz"
                local_tarpath = os.path.join(tmpdir, self._SRC_TAR_DIRNAME) + ".tar.gz"
                self.print_verbose(f"Downloading {remote_tarpath} to {local_tarpath}")
                fs.get(remote_tarpath, local_tarpath, callback=TqdmCallback())
                assert os.path.isfile(local_tarpath)
                self.print_verbose(f"Trying to parse local copy {local_tarpath}")
                _tardir = os.path.join(tmpdir, self._SRC_TAR_DIRNAME)
                with tarfile.open(local_tarpath, 'r') as _tarfile:
                    self.print_verbose(f"Extracting {local_tarpath} to {_tardir}")
                    _tarfile.extractall(tmpdir)
                self.print_debug(f"Extracted dir: {os.listdir(_tardir)}")
                logcounts_src_path = os.path.join(_tardir, self._SRC_DAT_FILENAME)
                topic_frame = logcounts_frame = pd.read_csv(logcounts_src_path, sep='\t', header=0, index_col=0).transpose()

                topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
                self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")


class miRCoHN(Datablock):
    """
        Data for the clustering HNSC study described in from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854517/.
        TODO: do not save 'pivots' or 'downregulated_mirna_infixes' to a file, return them from code instead?
    """
    REVISION = "1.2.1"

    @dataclass
    class SCOPE:
        logcounts: pd.DataFrame 
    
    TOPICS = {'logcounts': f"mircohn_rpm_log2.parquet",
              'counts': f"mircohn_rpm.parquet",
              'logcontrols': f"mircohn_logcontrols.parquet",
              'controls': f"mircohn_ontrols.parquet",
              'seq_patterns': f"seq_patterns.parquet",
              'pivots': f"mircohn_pivots.parquet",
    }

    SEQ_PATTERNS = dict(
        epithelial = list(set(['miR-150', 'miR-125b', 'miR-195', 'miR-127', 'miR-342', 'miR-361',
                                  'miR-195', 'miR-125b', 'miR-150', 'miR-149', 'miR-342'
               
        ])),
        stromal = list(set(['miR-210', 'miR-20a', 'miR-92a', 'miR-20b', 'miR-17', 'miR-200c', 'miR-200b', 
                                   'miR-200a', 'miR-425', 'miR-18a', 'miR-183', 'miR-224', 'miR-181d', 'miR-221', 'miR-93', 'miR-106b', 
                                   'miR-194', 'miR-660',
                                   'miR-25', 'miR-106b', 'miR-93', 'miR-92a', 'miR-17', 'miR-20a', 'miR-210', 'miR-200a', 'miR-200c', 
                                   'miR-200b', 'miR-194'
        ]))
    )

    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854517/bin/NIHMS1644540-supplement-3.docx
    PIVOT_SEQS = [
            "hsa-let-7d-5p",
            "hsa-miR-103a-3p",
            "hsa-miR-106a-5p",
            "hsa-miR-106b-3p",
            "hsa-miR-106b-5p",
            "hsa-miR-1180-3p",
            "hsa-miR-125b-5p",
            "hsa-miR-127-5p",
            "hsa-miR-1301-3p",
            "hsa-miR-1307-3p",
            "hsa-miR-149-5p",
            "hsa-miR-150-5p",
            "hsa-miR-151a-5p",
            "hsa-miR-17-3p",
            "hsa-miR-17-5p",
            "hsa-miR-181d-5p",
            "hsa-miR-182-5p",
            "hsa-miR-183-5p",
            "hsa-miR-18a-3p",
            "hsa-miR-194-5p",
            "hsa-miR-195-5p",
            "hsa-miR-200a-3p",
            "hsa-miR-200a-5p",
            "hsa-miR-200b-3p",
            "hsa-miR-200b-5p",
            "hsa-miR-200c-3p",
            "hsa-miR-205-5p",
            "hsa-miR-20a-5p",
            "hsa-miR-20b-5p",
            "hsa-miR-210-3p",
            "hsa-miR-221-3p",
            "hsa-miR-222-3p",
            "hsa-miR-224-5p",
            "hsa-miR-23b-5p",
            "hsa-miR-25-3p",
            "hsa-miR-27a-5p",
            "hsa-miR-27b-5p",
            "hsa-miR-320c",
            "hsa-miR-324-3p",
            "hsa-miR-342-5p",
            "hsa-miR-361-5p",
            "hsa-miR-369-5p",
            "hsa-miR-423-3p",
            "hsa-miR-425-5p",
            "hsa-miR-493-3p",
            "hsa-miR-660-5p",
            "hsa-miR-769-3p",
            "hsa-miR-92a-3p",
            "hsa-miR-93-3p",
            "hsa-miR-93-5p",
        ]

    #https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854517/bin/NIHMS1644540-supplement-4.docx
    @staticmethod
    def control_records_mask(counts):
        controls = pd.Series(counts.index, index=counts.index).apply(lambda _: _.split('-')[3].startswith('11'))
        return controls
    
    def display(
            self,
    ):
        cof = self.read()
        self.display_umap(cof)
        """
        ufit = umap.UMAP()
        ucof = ufit.fit_transform(cof.fillna(0.0))
        plt.scatter(ucof[:, 0], ucof[:, 1])
        """

    @staticmethod
    def seq_matches_patterns(seq, seq_patterns): 
            for pattern in seq_patterns:
                if seq.find(pattern) != -1:
                    return True
            return False

    @staticmethod
    def filter_columns_by_patterns(frame, col_patterns):
        if col_patterns is not None:
            fcols = [col for col in frame.columns.get_level_values(0) if miRCoHN.seq_matches_patterns(col, col_patterns)]
            fframe = frame[fcols]
        else:
            fframe = frame
        return fframe

    @staticmethod
    def filter_columns_by_mad(frame, mad_threshold):
        mads = (frame - frame.mean()).abs().mean()
        madf = pd.DataFrame({'mad': mads})
        madcuts = pd.qcut(madf.mad, 100, labels=False, duplicates='drop')
        madcols = madcuts[madcuts > mad_threshold].index
        madframe = frame[madcols]
        return madframe

    @staticmethod
    def center_at_controls(frame, controls):
        if controls is None:
            return frame
        cm = controls.mean(axis=0)
        cframe = frame - cm
        return cframe
    
    @staticmethod
    def expcounts(logcounts):
        counts = np.exp(logcounts.copy()*np.log(2))
        return counts
    
    @staticmethod
    def display_heatmap(counts:             pd.DataFrame, 
                         *, 
                         ordering:           Optional[List[int]] = None, 
                         seq_patterns:       Optional[List[str]] = None, 
                         seq_mad_threshold:  float = 0.0,
                         center_at_controls: Optional[pd.DataFrame] = None,
                         nseqs:              Optional[int] = None,
    ):
        """
            Inputs/Output: see 'normalized_counts()'
        """
        ncounts1 = miRCoHN.filter_columns_by_patterns(counts, seq_patterns)
        if len(ncounts1.columns) == 0:
            raise ValueError(f"'filter_columns_by_patterns left no columns")
        ncounts2 = miRCoHN.filter_columns_by_mad(ncounts1, seq_mad_threshold)
        if center_at_controls is not None:
            ncounts3 = miRCoHN.center_at_controls(ncounts2, center_at_controls[ncounts2.columns])
        else:
            ncounts3 = ncounts2
        ncounts = ncounts3.iloc[ordering]

        ncountst_ = ncounts.transpose()
        if nseqs is not None:
            ncountst = ncountst_.iloc[:nseqs]
        else:
            ncountst = ncountst_
        _ = px.imshow(ncountst.values, aspect='auto')
        return _

    def build(self):
        """
            Generate a pandas dataframe of TCGA HNSC mature MiRNA sequence samples.
        """
        framepaths = {topic: self.path(topic) for topic in self.TOPICS}
        if self.built():
            self.print_verbose("All topics built already.  Done.")
        else:
            self.print_verbose("Building ...")
            # logcounts
            topic = 'logcounts'
            topic_tgt_path = framepaths[topic]
            _logcounts_frame = self.scope.logcounts
            logcontrols_mask = miRCoHN.control_records_mask(_logcounts_frame)
            topic_frame = logcounts_frame = _logcounts_frame[~logcontrols_mask]

            coltuples = [tuple(c.split('|')) for c in logcounts_frame.columns]
            mindex = pd.MultiIndex.from_tuples(coltuples)
            logcounts_frame.columns = mindex
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            # counts
            topic = 'counts'
            topic_frame = counts_frame = self.expcounts(logcounts_frame)
            topic_tgt_path = framepaths[topic]
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            # pivots
            topic = 'pivots'
            topic_tgt_path = framepaths[topic]
            topic_frame = pivots_frame = pd.DataFrame({'pivots': self.PIVOT_SEQS})
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            #logcontrols
            topic = 'logcontrols'
            topic_tgt_path = framepaths[topic]
            topic_frame = logcontrols_frame = _logcounts_frame[logcontrols_mask]
            ccoltuples = [tuple(c.split('|')) for c in logcontrols_frame.columns]
            cmindex = pd.MultiIndex.from_tuples(ccoltuples)
            logcontrols_frame.columns = cmindex
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            #controls
            topic = 'controls'
            topic_tgt_path = framepaths[topic]
            topic_frame = controls_frame = self.expcounts(logcounts_frame)
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            #downregulated
            topic = 'seq_patterns'
            topic_tgt_path = framepaths[topic]
            epithelial_downregulated_infixes = self.SEQ_PATTERNS['epithelial']
            stromal_downregulated_infixes = self.SEQ_PATTERNS['stromal']
            topic_frame = downregulated_frame = pd.DataFrame.from_records([{'epithelial': ','.join(list(epithelial_downregulated_infixes)), 
                                                                            'stromal': ','.join(list(stromal_downregulated_infixes))}])
            topic_frame.to_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote dataframe to {topic_tgt_path}")

            self.print_verbose("... done")
        return framepaths
    
    def read(self, topic,):
        self.print_verbose(f"Reading topic '{topic}'")
        topic_tgt_path = self.path(topic)
        topic_frame = pd.read_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
        return topic_frame


class miRNA(Datablock):
    REVISION = "0.3.1"

    @dataclass
    class SCOPE:
        pass

    MIRNA_DATASET_URL = "https://mirbase.org/download"
    MIRNA_DATASET_FILENAME = f"miRNA"
    FILENAME = f"{MIRNA_DATASET_FILENAME}.parquet"
    
    def build(self):
        root = self.roots
        filesystem = self.filesystem
        if self.built():
            self.print_verbose(f"miRNA already built in root {root}")
        else:
            self.print_verbose(f"Building miRNA ...")

            fs = fsspec.filesystem('http')

            remote_dat = self.MIRNA_DATASET_URL + '/' + f'{self.MIRNA_DATASET_FILENAME}.dat'
            local_dat = os.path.join(root, f'{self.MIRNA_DATASET_FILENAME}.dat')
            if not os.path.isfile(local_dat):
                self.print_verbose(f"Downloading {remote_dat} to {local_dat}")
                fs.get(remote_dat, local_dat, callback=TqdmCallback())
            self.print_verbose(f"Parsing local copy {local_dat}")

            if local_dat.endswith('.gz'):
                with gzip.open(local_dat, 'r') as datfile:
                    datstr = datfile.read().decode()
                    frame = self._build_frame(datstr)
            else:
                with open(local_dat, 'r') as datfile:
                    datstr = datfile.read()
                    frame = self._build_frame(datstr)

            path = self.path()
            frame.to_parquet(path, storage_options=self.filesystem.storage_options)
            self.print_verbose(f"Wrote frame of len {len(frame)} to path")
            self.print_verbose("... done")

    def read(self):
        path = self.path()
        frame = pd.read_parquet(path, storage_options=self.filesystem.storage_options)
        return frame
    
    def _build_frame(self, mdstr):
        recs = miRNA._parse_records(mdstr)
        f = pd.DataFrame.from_records(recs)
        frame = f.sort_values('ID').reset_index(drop=True)
        self.print_verbose(f"Built dataframe")
        return frame

    @staticmethod     
    def _parse_records(mdstr):
        _mdstrs = mdstr.split('\nID')
        mdstrs = [f"ID{s}" for s in _mdstrs]
        _prerecs = [miRNA._parse_prerecord(s) for s in mdstrs]
        prerecs = [pr for pr in _prerecs if pr['DE'].find('sapiens') != -1]
        recs = [miRNA._prerecord_to_record(pr) for pr in prerecs]
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


class miRCoSeqs(Datablock):
    """
        Sequences sampled at count frequences
    """
    REVISION = "1.4.0"
    TOPICS = {'logcounts': f"miRLogCos.parquet",
                 'counts': f"miRCos.parquet",
                 'logcontrols': f"miRLogCtrls.parquet",
                 'controls': f"miRCtrls.parquet",
                 'seqs': f"miRSeqs.parquet",
                 'samples': f"miRCoSeqs.txt",
                 'rec_sample_ranges': f"miRSampleRanges.parquet"
    }
    
    @dataclass
    class SCOPE:
        seqs: pd.DataFrame
        logcounts: pd.DataFrame
        logcontrols: pd.DataFrame
        npasses: int = 5
        nseqs_per_record: int = 200    

    @staticmethod
    def count_cols_to_coseq_cols(columns):
        subcolumns = {c: c[1][5:] for c in columns}
        return subcolumns

    @staticmethod
    def expcounts(logcounts):
        counts = np.exp(logcounts.copy()*np.log(2))
        return counts
    
    def build(self):
        roots = self.roots
        filesystem = self.filesystem
        scope = self.scope
        def lift_jcols(frame, coseq2co):
            _cocols = frame.rename(columns=coseq2co).columns
            cocols = pd.MultiIndex.from_tuples(_cocols)
            frame.columns = cocols
            return frame

        if self.built():
            self.print_verbose(f"miRCoSeqs already built")
        else:
            self.print_verbose(f"Building miRCoSeqs ... ")
            logcof = scope.logcounts
            co2coseq = self.count_cols_to_coseq_cols(logcof.columns)
            coseq2co = {val: key for key, val in co2coseq.items()}
            _logcof = logcof.copy()
            _logcof.columns = list(co2coseq.values())

            # log2(n) -> n
            # n = exp(ln(n)) = exp[ln(2^log2(n))] = exp[log2(n)*ln(2)]
            _cof = self.expcounts(_logcof)

            seqf = scope.seqs
            accession = seqf.Accession.apply(lambda _: _[2:])
            _seqf = seqf.copy()
            _seqf['accession'] = accession
            _seqf.set_index('accession', inplace=True)

            # join columns/datasets
            jcols = [i for i in _seqf.index if i in _cof.columns]

            jcof = lift_jcols(_cof[jcols], coseq2co)
            jcof_path = self.path('counts')
            jcof.to_parquet(jcof_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote counts to {jcof_path}")

            jlogcof = lift_jcols(_logcof[jcols], coseq2co)
            jlogcof_path = self.path('logcounts')
            jlogcof.to_parquet(jlogcof_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote logcounts to {jlogcof_path}")
            
            jseqf = lift_jcols(_seqf.loc[jcols].transpose(), coseq2co).transpose()
            jseqs_path = self.path('seqs')
            jseqf.to_parquet(jseqs_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote sequences to {jseqs_path}")

            jcof0 = jcof.fillna(0.0)
            jcofn = jcof0.div(jcof0.sum(axis=1), axis=0)
            jseqs = jseqf['sequence']
            jseqlist = jseqs.tolist()

            jlogcontrols = scope.logcontrols[co2coseq.keys()]
            jlogcontrols_path = self.path('logcontrols')
            jlogcontrols.to_parquet(jlogcontrols_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote logcontrols to {jlogcontrols_path}")

            jcontrols = self.expcounts(scope.logcontrols[co2coseq.keys()])
            jcontrols_path = self.path('controls')
            jcontrols.to_parquet(jcontrols_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote controls to {jcontrols_path}")

            self.print_verbose(f"Generating samples using {scope.npasses} passes")
            rec_sample_ranges = {recidx: [] for recidx in jcofn.index}
            sample_batches = []
            n_samples = 0
            rng = np.random.default_rng()
            for _pass in range(scope.npasses):
                self.print_verbose(f"pass {_pass}")
                n_pass_samples = 0
                for recidx, rec in jcofn.iterrows():
                    rec_sample_batches = []
                    n_rec_samples = 0
                    seqfreqs = rng.multinomial(scope.nseqs_per_record, rec)
                    for i, freq in enumerate(seqfreqs):
                        if freq == 0: continue
                        rec_sample_batches.append((i, freq))#jseqlist[i:i+1]*freq
                        n_rec_samples += freq
                    rec_sample_ranges[recidx].append((n_samples, n_samples + n_rec_samples))
                    sample_batches += rec_sample_batches
                    n_samples += n_rec_samples
                    n_pass_samples += n_rec_samples
                self.print_verbose(f"Generated {n_pass_samples} in pass {_pass} for a total of {n_samples} so far")
            self.print_verbose(f"Generated {n_samples} samples")

            samples_path = self.path('samples')
            with filesystem.open(samples_path, 'w') as f:
                self.print_verbose(f"Writing {n_samples} samples to {samples_path}")
                for i, freq in sample_batches:
                    batch = jseqlist[i:i+1]*freq
                    s = "\n".join(batch)+"\n"
                    f.write(s)

            rec_sample_ranges_frame = pd.DataFrame(rec_sample_ranges).transpose()
            rec_sample_ranges_frame.columns.name = 'pass'
            rec_sample_ranges_path = self.path('rec_sample_ranges')
            rec_sample_ranges_frame.to_parquet(rec_sample_ranges_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote {len(rec_sample_ranges_frame)} to {rec_sample_ranges_path}")
        self.print_verbose("... done")

    def read(self, topic):
        roots = self.roots
        filesystem = self.filesystem
        path = self.path(topic)
        if topic == 'samples':
            with filesystem.open(path, 'r') as f:
                s = f.read()
            self.print_debug(f"Read string of len {len(s)} from {path}")
            useqs = s.split('\n')
            self.print_verbose(f"Read {len(useqs)} useqs")
            _ = useqs
        elif topic in self.TOPICS:
            _ = pd.read_parquet(path, storage_options=filesystem.storage_options)
        else:
            raise ValueError(f"Unknown topic: {topic}")
        return _
    
    def display(self, topic):
        if topic == 'samples':
            ...
        elif topic == 'logcounts': 
            ...
        elif topic == 'counts': 
            ...
        elif topic == 'seqs':
            ...


class ZSCC(Datablock):
    REVISION = "0.8.1"
    TOPICS = {
        'zscc': 'zscc.pkl',
        'clusters': 'clusters.parquet',
        'ordering': 'ordering.pkl'
    }
    
    @dataclass
    class SCOPE:
        """
            * `clustering` is a Callable that takes `n_clusters: int`
            * `clustering(n).fit_predict(y: np.array(m, n)) -> np.array(m)` returns cluster assignments
        """
        data_frame: pd.DataFrame
        clustering: str = 'sklearn.cluster.KMeans'
        n_reps: int = 100
        lo: int = 2
        hi: int = 5
        fillna: Optional[float] = None

    def build(self,):
        scope = self.scope
        filesystem = self.filesystem
        if self.built():
            self.print_verbose(f"ZSCC already built")
        else:
            self.print_verbose("Building ZSCC ...")
            clparts = scope.clustering.split('.')
            clmodname = '.'.join(clparts[:-1])
            clclassname = clparts[-1]
            clmod = importlib.import_module(clmodname)
            clustering = getattr(clmod, clclassname)

            zscc = ZSConsensusClustering(clustering, n_reps=scope.n_reps, lo=scope.lo, hi=scope.hi)
            data_frame = scope.data_frame if scope.fillna is None else scope.data_frame.fillna(scope.fillna)
            self.print_verbose(f"Fitting zscc to data of size {len(data_frame)}")
            zscc.fit(data_frame.values)

            zscc_path = self.path('zscc')
            with filesystem.open(zscc_path, 'wb') as zsccfile:
                pickle.dump(zscc, zsccfile)
            if self.verbose:
                print(f"Wrote zscc pickle to {zscc_path}")

            if self.verbose:
                print(f"Assigning optimal clusters to data of len {len(data_frame)}")
            clusters_path = self.path('clusters')
            clusters = zscc.predict_data(data_frame.values)
            clusters_frame = pd.DataFrame({'clusters': clusters})
            clusters_frame.to_parquet(clusters_path, storage_options=filesystem.storage_options)
            self.print_verbose(f"Wrote zscc clusters to {clusters_path}")

            ordering_path = self.path('ordering')
            ordering = np.argsort(clusters)
            with filesystem.open(ordering_path, 'wb') as ordering_file:
                pickle.dump(ordering, ordering_file)
            self.print_verbose(f"Wrote zscc ordering to {ordering_path}")
        self.print_verbose("... done")

    def read(self, topic):
        roots = self.roots
        filesystem = self.filesystem
        if topic == 'zscc':
            zscc_path = self.path('zscc')
            with filesystem.open(zscc_path, 'rb') as zsccfile:
                zscc = pickle.load(zsccfile)
                _ = zscc
            if self.verbose:
                print(f"Loaded zscc pickle from {zscc_path}")
        elif topic == 'clusters':
            clusters_path = self.path('clusters')
            clusters_frame = pd.read_parquet(clusters_path, storage_options=filesystem.storage_options)
            _ = clusters_frame
            if self.verbose:
                print(f"Read zscc clusters from {clusters_path}")
        elif topic == 'ordering':
            ordering_path = self.path('ordering')
            with filesystem.open(ordering_path, 'rb') as ordering_file:
                _ = pickle.load(ordering_file)
            if self.verbose:
                print(f"Read zscc cluster ordering from {ordering_path}")
        else:
            raise ValueError(f"Unknown topic '{topic}'")
        return _


class FastText(Datablock):
    REVISION = "0.8.1"
    
    @dataclass
    class SCOPE:
        '''
        fasttext manpage:
            input             # training file path (required)
            model             # unsupervised fasttext model {cbow, skipgram} [skipgram]
            lr                # learning rate [0.05]
            dim               # size of word vectors [100]
            ws                # size of the context window [5]
            epoch             # number of epochs [5]
            minCount          # minimal number of word occurences [5]
            minn              # min length of char ngram [3]
            maxn              # max length of char ngram [6]
            neg               # number of negatives sampled [5]
            wordNgrams        # max length of word ngram [1]
            loss              # loss function {ns, hs, softmax, ova} [ns]
            bucket            # number of buckets [2000000]
            thread            # number of threads [number of cpus]
            lrUpdateRate      # change the rate of updates for the learning rate [100]
            t                 # sampling threshold [0.0001]
            verbose           # verbose [2]
        '''
        samples_path: str
        model: str = "cbow"
        dim: int = 100
        context_window_size: int = 100

    FILENAME = "model.bin"
    
    def build(self):
        scope = self.scope
        root = self.roots
        filesystem = self.filesystem
        if self.built():
            self.print_verbose(f"'{scope.model}' already built")
        else:
            self.print_verbose(f"Building '{scope.model}' ...")
            path = self.path()
            if not filesystem.exists(path):
                samples_files = filesystem.ls(scope.samples_path)
                assert len(samples_files) == 1
                samples_path = samples_files[0]
                cbow = fasttext.train_unsupervised(samples_path, model=scope.model, dim=scope.dim, ws=scope.context_window_size)
                cbow.save_model(path)
            self.print_verbose("... done")

    def read(self,):
        path = self.path()
        model = fasttext.load_model(path)
        return model


class HNSCCProfiles(Datablock):
    """
        HNSCC records of expression counts (miR, mR) and HPV status
    """
    REVISION = '0.1.1'
    TOPICS = ['mir_co',
              'mr_co',
              'hpv_status',
              'full',
              'mr_co_gene_ids',
              ]

    @dataclass
    class SCOPE:
        srcpath: str = None

    def __setup__(self):
        if self.scope.srcpath is None:
            filepath = inspect.getmodule(self.__class__).__file__
            # .../quantum/micron/micron/datablocks.py
            qroot= os.path.dirname(os.path.dirname(os.path.dirname(filepath)))

            self.srcpath = qroot + '/' + '/'.join(('data', 'HNSCC', 'full'))
            
            if self.filesystem.protocol != "file":
                raise 
        else:
            self.srcpath = self.scope.srcpath

    @functools.lru_cache(maxsize=None)
    def read(self, topic):
        if topic not in self.TOPICS:
            raise ValueError(f"Topic '{topic}' not among known topics {self.TOPICS}")
        
        if topic == 'mr_co':
            filepath = f"{self.srcpath}/tcga_hnscc_all_sig_mrna_full.csv"
            _frame = pd.read_csv(filepath, index_col=0, storage_options=self.filesystem.storage_options)
            columns = [c for c in _frame.columns if c.find('|') != -1 and c.split('|')[0] != '?']
            frame = _frame[columns]
        elif topic == 'mir_co':
            filepath = f"{self.srcpath}/tcga_hnscc_all_mirna_full.csv"
            frame = pd.read_csv(filepath, index_col=0, storage_options=self.filesystem.storage_options)
        elif topic == 'hpv_status':
            filepath = f"{self.srcpath}/combined_full.csv"
            _frame = pd.read_csv(filepath, index_col=0, storage_options=self.filesystem.storage_options)
            frame = _frame[['HPV_status']]
        elif topic == 'mr_co_gene_ids':
            coframe = self.read("mr_co")
            gene_ids = [c.split('|')[1] for c in coframe.columns]
            self.print_debug(f"DEBUG: >>>>>>>>>>>> HNSCCProfiles: gene_ids: len(gene_ids): {len(gene_ids)}")
            frame = pd.DataFrame({'gene_ids': gene_ids}, index=gene_ids).transpose()
        elif topic == 'full':
            filepath = f"{self.srcpath}/combined_full.csv"
            frame = pd.read_csv(filepath, index_col=0, storage_options=self.filesystem.storage_options)
        return frame
    
    def valid(self, topic):
        return True


class HNSCCmRCoHPV(Datablock):
    """
        HNSCC: unique mRNA count records that have a matching HPV status
    """
    REVISION = '0.1.1'
    TOPICS = {
              'mrco': 'mRCo.parquet',
              'hpv': 'hpv.parquet',
    }

    @dataclass
    class SCOPE:
        mr_co: pd.DataFrame
        hpv_status: pd.DataFrame

    def _write_dataframe(self, topic, frame):
        path = self.path(topic)
        frame.to_parquet(path, storage_options=self.filesystem.storage_options)
        self.print_verbose(f"Wrote {self.__class__.__qualname__} '{topic}' frame with {len(frame)} rows to {path}")

    def read(self, topic):
        path = self.path(topic)
        frame = pd.read_parquet(path, storage_options=self.filesystem.storage_options)
        self.print_verbose(f"Read {self.__class__.__qualname__} '{topic}' frame with {len(frame)} rows from {path}")
        return frame

    def build(self):
        if self.built():
            self.print_verbose(f"{self.__class__.__name__} already built")
        else:
            self.print_verbose(f"Building {self.__class__.__name__} ...")
            _hmrco = self.scope.mr_co
            _hmrco.index.name = 'patient_id' 
            _hmrco_unique = _hmrco.groupby('patient_id').std().isnull().all(axis=1) # patients with no duplicates
            hmrco = _hmrco[_hmrco_unique]
            
            hs = self.scope.hpv_status.groupby('patient_id').first()
            jhmrcos = hmrco.join(hs, how='inner', validate="1:1")

            jhmrco = jhmrcos[[c for c in jhmrcos.columns if c != 'HPV_status']]
            jhs = jhmrcos[['HPV_status']].rename(columns={'HPV_status': 'hpv'})

            #hpv
            self._write_dataframe('hpv', jhs)
            
            #mrco
            self._write_dataframe('mrco', jhmrco)

            self.print_verbose("... done")

#TODO: FIX
'''
class mRSeqs(Datablock):
    """
        mRNA sequences for a given set of gene_ids.
        gene_id: gene_symbol|ncbi_id
        
    """

    REVISION = '0.2.1'
    TYPE = pd.DataFrame
    SCHEMA = {"gene_id": str, "seq": str}

    class RANGE(tuple):
        lo: int
        hi: Optional[int] = None

        def __post_init__(self):
            if self.high is None:
                self.high = self.lo + 1

    @dataclass
    class SCOPE:
        gene_ids_frame: pd.DataFrame
        gene_ids_bucket: RANGE

    def path(self, gene_id):
        sym, ncbi = gene_id.split('|')
        filename = f"mrseqs.{sym}.{ncbi}"
        if self.filesystem.protocol == 'file':
            if self.roots is None:
                path = os.join(os.getcwd(), filename)
            else:
                path_ = self.roots
                os.makedirs(path_, exist_ok=True)
                path = os.path.join(path_, filename)
        else:
            path = self.roots + "/" + filename
        return path
    
    def gene_ids(self):
        return list(self.scope.gene_id_cols.columns)

    def gene_ids_bucket(self):
        lo, hi = self.scope.gene_ids_bucket.lo, self.scope.gene_ids_bucket.hi
        bucket = self.gene_ids[lo:hi]
        return bucket

    def paths(self):
        paths = {}
        for gene_id in self.gene_ids_bucket:
            path = self.path(gene_id)
            if path is not None:
                paths[gene_id] = path
        return _

    def build(self):
        # gene_ids are of the form: "{gene_symbol}|{ncbi_id}"; we need gene_symbols.
        # for example, 'ABCD2|225' -> 'ABCD2'
        # ncbi_id may also be known as Entrez ID or simply Entrez.
        self.print_verbose(f"{self.__class__.__name__}: starting build of batch of bucket {self.sccope.bucket} " + 
                           f"with gene_ids {len(self.gene_ids)} gene_ids ...")

        #TODO: 1. Store all sequences from the bucket in the same file. 
        #      2. Try all ids in the request first.
        #      3. Fall back on individual requests in case of failure
        paths = self.paths()
        failed_gene_ids = []
        for gene_id, path in paths.items():
            try:
                seq = self._fetch_mrna_sequences([gene_id])
                frame = pd.DataFrame({k: str(v) for k, v in seqs.items()}, index=['seq']).transpose()
                frame.to_parquet(path, storage_options=self.filesystem.storage_options)
                self.print_verbose(f"{self.__class__.__name__}: wrote frame of len {len(frame)} to {path}")
            except:
                failed_gene_ids.append(gene_id)
        if len(failed_gene_ids) == 0:
            self.print_verbose(f"{self.__class__.__name__}: build complete")
        else:
            raise ValueError(f"Failed gene_ids: {failed_gene_ids}")

    def read(self):
        paths = self.paths()
        frame = pd.read_parquet(list(paths.values()), storage_options=self.filesystem.storage_options)
        self.print_verbose(f"{self.__class__.__name__}: read frame of len {len(frame)} from {len(paths)} paths")
        return frame
    
    def valid(self):
        metric = self.metric()
        valid = metric is not None
        return valie
        
    def metric(self):
        try:
            frame = self.read()
            _ = len(frame) if frame is not None else 0
        except:
            _ = None
        return _

    @staticmethod
    def _fetch_mrna_sequences(gene_ids):
        """
        Fetch mRNA sequences for given gene symbols from NCBI.

        :param gene_symbols: A list of gene symbols.
        :return: A dictionary with gene symbols as keys and mRNA sequences as values.
        """

        gene_symbols = [c.split('|')[0] for c in gene_ids]
        sequences = {}
        for i, symbol in enumerate(gene_symbols):
            # Search for the gene symbol in the nucleotide database to get the sequence IDs
            handle = Entrez.esearch(db="nucleotide", term=f"{symbol}[Gene Name] AND mRNA[Filter]", retmax=10)
            record = Entrez.read(handle)
            handle.close()

            # Fetch the sequences for each ID found
            for seq_id in record["IdList"]:
                handle = Entrez.efetch(db="nucleotide", id=seq_id, rettype="fasta", retmode="text")
                seq_record = SeqIO.read(handle, "fasta")
                handle.close()

                # Store the sequence
                sequences[gene_ids[i]] = seq_record.seq

        return sequences
'''
    
class GRCh38(Datablock):
    """
        GRCh38: Genome Reference Consortium Human Build 38 Organism: Homo sapiens (human) Submitter: Genome Reference Consortium Date: 2013/12/17

        Docs:
            * GRCh38: https://gatk.broadinstitute.org/hc/en-us/articles/360035890951-Human-genome-reference-builds-GRCh38-or-hg38-b37-hg19
            * GFF format: 
                . https://learn.gencore.bio.nyu.edu/ngs-file-formats/gff3-format/
                . https://github.com/The-Sequence-Ontology/Specifications/blob/master/gff3.md
            * FNA format: FASTA nucleic acid
                . https://en.wikipedia.org/wiki/FASTA_format
        Tools:
            * Genome browser: https://genome.ucsc.edu/cgi-bin/hgGateway
    """
    REVISION = "0.0.1"
    TOPICS = {'sequences': "GRCh38_latest_genomic.sequences.parquet",
              'annotations': "GRCh38_latest_genomic.annotations.parquet",
              'attributes': None,
    }
    
    GRCh38_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/H_sapiens/annotation/GRCh38_latest/refseq_identifiers"
    GRCh38_FILE_BASENAME = "GRCh38_latest_genomic"
    EXTENSIONS = {'sequences': '.fna.gz',
                  'annotations': '.gff.gz'
    }

    def valid(self, topic):
        if topic == 'attributes':
            topic = "annotations"
        return super().valid(topic)

    def build(self):
        self.print_verbose(f"Building GRCh38")
        if self.filesystem.protocol != 'file':
            raise ValueError(f"Only local filesystem supported, instead got {self.filesystem}")
        
        for topic, ext in self.EXTENSIONS.items():
            self.print_verbose(f"Downloading data for topic {repr(topic)}")
            filename = self.GRCh38_FILE_BASENAME + ext
            httpfs = fsspec.filesystem('http')
            remote_fna = self.GRCh38_URL+'/'+f'{filename}'
            local_fna = self.path(topic)
            self.print_debug(f"Using files: {remote_fna} -> {local_fna}")
            self.print_verbose(f"Downloading {remote_fna} to {local_fna}")
            httpfs.get(remote_fna, local_fna, callback=TqdmCallback())
            self.print_verbose(f"Done downloading data for topic {repr(topic)}")

            with gzip.open(local_fna, 'r') as topicfile:
                self.print_verbose(f"Parsing data for topic {repr(topic)}")
                topicstr = topicfile.read().decode()
                if topic == 'sequences':
                    topicf = self._parse_sequences(topicstr)      
                if topic == 'annotations':
                    topicf = self._parse_annotations(topicstr)
                self.print_verbose(f"Done parsing data for topic {repr(topic)}")
                topicpath = self.path(topic)
                topicf.to_parquet(topicpath, storage_options=self.filesystem.storage_options)
                self.print_verbose(f"Stored frame with {len(topicf)} rows to {repr(topicpath)}")

    def read(self, topic):
        if topic != 'attributes':
            self.print_verbose(f"Reading '{self.__class__.__qualname__}' topic {repr(topic)}")
            topic_tgt_path = self.path(topic)
            topic_frame = pd.read_parquet(topic_tgt_path, storage_options=self.filesystem.storage_options)
        else:
            anns = self.read('annotations')
            topic_frame = pd.DataFrame({'attributes': [{key: val for key, val in [tuple(s.split('=')) for s in ann.attributes.split(";")]} for _, ann in anns.iterrows()]}, index=anns.index)
        return topic_frame

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
    def _parse_annotations(gffstr):
        gffstrs_ = gffstr.split('\n')
        gffstrs = [s for s in gffstrs_ if not s.startswith('#')]
        
        gffio = StringIO('\n'.join(gffstrs))
        gff = pd.read_csv(gffio, sep='\t', names=['seqid', 'source', 'type', 'start', 'end', 'score', 'strand', 'phase', 'attributes'])
        attr = gff.attributes.str.strip()
        gff['ID'] = attr.apply(lambda _: _[3:].split(';')[0].split(':')[0])
        return gff
    

class GRCh38GeneSeqs(Datablock):
    """
        Evaluation set for expression count models, such as Enformer
            * TODO: Enformer URL and literature
        We extract sequences following the procedure outlined in 
            * "Benchmarking of deep neural networks for predicting personal gene expression from DNA sequence highlights shortcomings", A. Sasse et al.
            * 'Methods. Predicting gene expressions with Enformer'
            * https://www.nature.com/articles/s41588-023-01524-6
    """
    REVISION = "0.0.1"
    FILENAME = "sequences.parquet"

    @dataclass
    class SCOPE:
        sequences: pd.DataFrame
        annotations: pd.DataFrame
        gene_ids: pd.DataFrame # row-vector of gene_ids (single row with columns indexing gene_ids)
        gene_bucket_number: int = 0
        gene_bucket_count: int = 100
        lookback_bps: int = 192000
        
    def build(self):
        assert (0 <= self.scope.gene_bucket_number) and \
               (self.scope.gene_bucket_number < self.scope.gene_bucket_count)
        N = len(self.scope.gene_ids)
        bucket_size = (N+1)//self.scope.gene_bucket_count
        lo = bucket_size*self.scope.bucket_number
        hi = min(N, lo+bucket_size)
        gene_ids = self.scope.gene_ids.iloc[lo:hi]
        genes = [g.split('|')[1] for g in gene_ids]

        path = self.path()


    
        

            

