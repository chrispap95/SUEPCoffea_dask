#############################################################################
# MERGE
#############################################################################
# Merge each of the chunks' .hdf5 files together
# This won't be triggered unless the SUEP processor runs smoothly through
# all the chunks, thus assuring we processed all the events
# N.B.: Only merging df named 'vars' in the HDF5 object

import pandas as pd
import os, sys
import glob
import h5py
import numpy as np

def h5load(ifile, label):
    try:
        with pd.HDFStore(ifile, 'r') as store:
            try:
                data = store[label] 
                metadata = store.get_storer(label).attrs.metadata
                return data, metadata
        
            except KeyError:
                print("No key",label,ifile)
                return 0, 0
    except:
        print("Some error occurred", ifile)
        return 0, 0

def merge(options): 
    files = glob.glob("condor_*.hdf5")
    if len(files) == 0: 
        print("No .hdf5 files found")
        sys.exit()
    
    df_tot = None
    metadata_tot = None
    for ifile, file in enumerate(files):
        
        df, metadata = h5load(file, 'vars') 
        
        ### Error out here
        if type(df) == int: 
            print("Something screwed up.")
            sys.exit()
            
        ### MERGE METADATA
        if metadata_tot is None: metadata_tot = metadata
        elif options.isMC: metadata_tot['gensumweight'] += metadata['gensumweight']
            
        # no need to add empty ones
        if 'empty' in list(df.keys()): continue
        
        ### MERGE DF
        if df_tot is None: df_tot = df
        else: df_tot = pd.concat((df_tot, df))   
        
    # SAVE OUTPUTS
    if df_tot is None: 
        print("No events in df_tot.")
        df_tot = pd.DataFrame(['empty'], columns=['empty'])
    store = pd.HDFStore("out.hdf5")
    store.put('vars', df_tot)
    store.get_storer('vars').attrs.metadata = metadata_tot
    store.close()
    
    # clean up the chunk files that we have already merged together
    for file in files:
        os.system("rm " + str(file))
    return


def merge_ML(options):
    
    files = glob.glob("*Events*.hdf5")
    
    # debug
    print(files)
    
    # skip if no files
    if len(files) == 0: 
        print("No .hdf5 files found")
        sys.exit()
    
    output = {}
    for ifile, file in enumerate(files):
        f = h5py.File(file, 'r')

        # skip if empty
        if 'empty' in list(f.keys()): 
            f.close()
            continue
        
        for key in f.keys():
            data = f[key]
            data = data[:]

            if ifile == 0:
                output[key] = data
            else:
                output[key] = np.vstack((output[key], data))

        f.close()
        
    # save to file
    outFile = "out.hdf5"
    with h5py.File(outFile, 'w') as outFile:
        for key, item in output.items():
            
            # debug
            print(key, item.shape)
            
            outFile.create_dataset(key, data=item, compression='gzip')
        
    # clean up the chunk files that we have already merged together
    for file in files:
        os.system("rm " + str(file))
    
    return