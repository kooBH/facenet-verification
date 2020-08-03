import os
import numpy as np
import itertools
import torch
import multiprocessing as mp

def genSameFile(args):
    global dict_filedata
    a,b =args
    t =np.abs(dict_filedata[a]-dict_filedata[b])
    np.save(dir_same +a[0:-8]+'_'+b[0:-8] ,t)

def genDiffFile(args):
    global dict_filedata
    a,b =args
    t = np.abs(dict_filedata[a]-dict_filedata[b])
    np.save(dir_diff +a[0:-8]+'_'+b[0:-8] ,t)


if __name__ == "__main__":

    dict_filename=dict()
    dict_filedata=dict()

    dir_orig = "/home/nas/user/kbh/FaceVerification/original/";
    dir_same= "data/same/";
    dir_diff= "data/diff/";

    filenames = os.listdir(dir_orig);
    num_core = mp.cpu_count()

    for i in filenames :
        key = i[0:3]
        if key in dict_filename.keys():
            dict_filename[key].append(i)
        else : 
            dict_filename.update({key:[i]})

    # open files
    for k,v in dict_filename.items():
        for vv in v:
            t_file = open(dir_orig + vv, 'rb')
            dict_filedata[vv]=np.fromfile(t_file,np.float32)

    # for same face
    #for c in itertools.combinations(dict_filename['kbh'],2) :
    #    t = dict_filedata[c[0]]-dict_filedata[c[1]]
    #    np.save(dir_same +c[0][0:5]+'_'+c[1][0:5] ,t)
    for name  in dict_filename :
        print('same : ' + name)
        pool_same = mp.Pool(processes=num_core)
        pool_same.map(genSameFile,itertools.combinations(dict_filename[name],2),chunksize=256)
        pool_same.close()
        pool_same.join()

    # for different faces
    # for a in dict_filename['kbh']:
    #     for b in dict_filename['nsh']:
    #         t = dict_filedata[a][0:5]-dict_filedata[b][0:5]
    #         np.save(dir_diff ++a+'_'+b ,t)
    #         #print(a + ' and ' + b)
    for comb in itertools.combinations(dict_filename,2):
        print('diff : ' + comb[0] +' vs '+comb[1] )
        pool_diff = mp.Pool(processes=num_core)
        pool_diff.map(genDiffFile,itertools.product(dict_filename[comb[0]],dict_filename[comb[1]]),chunksize=256)
        pool_diff.close()
        pool_diff.join()


