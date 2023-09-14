from dataclasses import dataclass

import os

import fsspec
from fsspec.callbacks import TqdmCallback
import tarfile
import tempfile

import pandas as pd
import pyarrow.parquet as pq

import ray


class Dataset:
    @dataclass
    class SCOPE:
        pass
        
    def __init__(self, debug: bool):
        self.debug = debug

    def build(self, 
             scope: SCOPE=SCOPE(), 
             filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"),
             root:str=os.getcwd(), 
        ):
        raise NotImplementedError()
        
    def read(self, 
             scope: SCOPE=SCOPE(), 
             filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"),
             root:str=os.getcwd(), 
        ):
        raise NotImplementedError()

    def valid(self, 
             scope: SCOPE=SCOPE(), 
             filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"),
             root:str=os.getcwd(), 
        ):
        raise NotImplementedError()
    
    def metric(self, 
             scope: SCOPE=SCOPE(), 
             filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"),
             root:str=os.getcwd(), 
        ):
        raise NotImplementedError()



class HNSCCountsMiRNA(Dataset):
    """
        Data for the clustering HNSC study described in from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7854517/.
    """
    VERSION = "0.0.1"
    MIRNA_SRC_URL = "https://gdac.broadinstitute.org/runs/stddata__2016_01_28/data/HNSC/20160128/"
    MIRNA_SRC_TAR_DIRNAME = "gdac.broadinstitute.org_HNSC.miRseq_Mature_Preprocess.Level_3.2016012800.0.0"
    MIRNA_SRC_DAT_FILENAME = "HNSC.miRseq_mature_RPM_log2.txt"
    MIRNA_TGT_DAT_FILENAME = f"mirna_rpm_log2.parquet"

    def __init__(self, debug=False, verbose=False, rm_tmp=True, ):
        super().__init__(debug)
        self.verbose = verbose
        self.rm_tmp = rm_tmp
    
    def build(self, scope: Dataset.SCOPE=Dataset.SCOPE(), filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"), root: str=os.getcwd()):
        """
            Generate a pandas dataframe of TCGA HNSC mature MiRNA sequence samples.
        """
        if self.verbose:
            print("Building HNSCCountsMiRNA")
        fs = fsspec.filesystem('http')
        with tempfile.TemporaryDirectory() as tmpdir:
            remote_tarpath = self.MIRNA_SRC_URL + '/' + self.MIRNA_SRC_TAR_DIRNAME + ".tar.gz"
            local_tarpath = os.path.join(tmpdir, self.MIRNA_SRC_TAR_DIRNAME) + ".tar.gz"
            if self.verbose:
                print(f"Downloading {remote_tarpath} to {local_tarpath}")
            fs.get(remote_tarpath, local_tarpath, callback=TqdmCallback())
            assert os.path.isfile(local_tarpath)
            if self.verbose:
                print(f"Trying to parse local copy {local_tarpath}")
            _tardir = os.path.join(tmpdir, self.MIRNA_SRC_TAR_DIRNAME)
            with tarfile.open(local_tarpath, 'r') as _tarfile:
                if self.verbose:
                    print(f"Extracting {local_tarpath} to {_tardir}")
                _tarfile.extractall(tmpdir)
            if self.debug:
                print(f"DEBUG: extracted dir: {os.listdir(_tardir)}")
            _datpath = os.path.join(_tardir, self.MIRNA_SRC_DAT_FILENAME)
            frame = pd.read_csv(_datpath, sep='\t', header=0, index_col=0)
            frame_path = root + "/" + self.MIRNA_TGT_DAT_FILENAME       
            frame.to_parquet(frame_path, storage_options=filesystem.storage_options)
            if self.verbose:
                print(f"Wrote dataframe to {frame_path}")
            return frame_path
    
    def read(self, scope: Dataset.SCOPE=Dataset.SCOPE(), filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"), root: str=os.getcwd()):
        frame_path = os.path.join(root, self.MIRNA_TGT_DAT_FILENAME)        
        frame = pd.read_parquet(frame_path, storage_options=filesystem.storage_options)
        return frame
    
    def valid(self, scope: Dataset.SCOPE=Dataset.SCOPE(), filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"), root: str=os.getcwd()):
        frame_path = os.path.join(root, self.MIRNA_TGT_DAT_FILENAME)        
        valid = filesystem.isfile(frame_path)
        return valid

    def metric(self, scope: Dataset.SCOPE=Dataset.SCOPE(), filesystem: fsspec.AbstractFileSystem=fsspec.filesystem("file"), root: str=os.getcwd()):
        frame_path = os.path.join(root, self.MIRNA_TGT_DAT_FILENAME)        
        valid = filesystem.isfile(frame_path)
        return int(valid)