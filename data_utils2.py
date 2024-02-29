import torch
import os
import re
import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count

#import sys

def directory_spider(input_dir, path_pattern="", file_pattern="", maxResults=500):
    file_paths = []
    if not os.path.exists(input_dir):
        raise FileNotFoundError("Could not find path: %s"%(input_dir))
    for dirpath, dirnames, filenames in os.walk(input_dir):
        if re.search(path_pattern, dirpath):
            file_list = [item for item in filenames if re.search(file_pattern,item)]
            file_path_list = [os.path.join(dirpath, item) for item in file_list]
            file_paths += file_path_list
            if len(file_paths) > maxResults:
                break
    return file_paths[0:maxResults]



class EEG_dataset_flexible(torch.utils.data.Dataset):
    def __init__(self, data_path, beforePts,afterPts,targetPts, channelIdxs=1, transform=None,preprocess=False,limit=None):
        self.transform = transform
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.channelIdxs=channelIdxs
        self.nChannels=len(channelIdxs) if isinstance(channelIdxs, (list,tuple,range)) else 1
        self.file_paths=directory_spider(data_path, path_pattern="", file_pattern=".set", maxResults=500)
        
        self.limit=limit

        self.preLoad()

    def preLoad(self):
        maxFilesLoaded=self.determineMemoryCapacity()

        #preload:
        self.raws=[]
        nfilesToLoad=min(maxFilesLoaded,len(self.file_paths))
        fileIdxToLoad=np.random.choice(len(self.file_paths),nfilesToLoad,replace=False)
        for fileIdx in fileIdxToLoad:
            tempRaw=mne.io.read_raw_eeglab(self.file_paths[fileIdx],preload=True)

            channelsToExclude=(1- np.isin(range(0,tempRaw.info['nchan']),self.channelIdxs)).nonzero()[0].astype('int')
            # import pdb; pdb.set_trace()
            channelsToExclude=np.asarray(tempRaw.ch_names)[channelsToExclude]
            tempRaw.drop_channels(channelsToExclude)
            self.raws.append(tempRaw)
        
        if self.limit:
            self.dataDraws=np.zeros((self.__len__(),3),np.int64) #columns for: file, channel, time
            print('Preparing ready-made data draws...')


            def myfun(arg):
                result=self.getAllowedDatapoint()
                return result

            results = Parallel(n_jobs=np.max((1,cpu_count()-1)), verbose=1, backend="threading")(map(delayed(myfun), range(self.__len__())))

            self.dataDraws=np.asarray(results)

            # for i in range(self.__len__()):

            #     randFileIdx,channelIdx,randomIdx=self.getAllowedDatapoint()
                
            #     self.dataDraws[i,0]=randFileIdx
            #     self.dataDraws[i,1]=channelIdx
            #     self.dataDraws[i,2]=randomIdx

    def determineMemoryCapacity(self):
        #determine how much space we can use for pre-loaded data:
        import psutil
        freeMemory=psutil.virtual_memory().available
        print("Detected free memory:",freeMemory / (1024**3),"GB")

        fileSizeMax=10*3600*250 #10 hours of data at 250Hz
        fileSizeMax=fileSizeMax*self.nChannels
        fileSizeMax*=64/8 #size of a 10 hr night in bytes

        nFiles=int(freeMemory/fileSizeMax)
        print("This will fit approximately %s files with %s  channels each"%(nFiles,self.nChannels))
        print('')

        return nFiles

    def getAllowedDatapoint(self,returnData=False):
        windowSize=self.beforePts+self.afterPts+self.targetPts
        #keep looking until we find a data window without nan's
        data=np.nan


        while np.any(np.isnan(data)):            
            randFileIdx=np.random.randint(0,len(self.raws))    
            randomChannelIdx=np.random.choice(self.nChannels)
            randomIdx=np.random.randint(0,self.raws[randFileIdx].n_times-windowSize)

            data,_=self.raws[randFileIdx][randomChannelIdx,randomIdx:randomIdx+windowSize]


        if returnData:
            return randFileIdx,randomChannelIdx,randomIdx,data
        else:
            return randFileIdx,randomChannelIdx,randomIdx


    def __len__(self):
        if self.limit:
            numel=self.limit
        else:
            # numel=int(np.sum([raw.n_times for raw in self.raws])*self.nChannels/2)
            numel=100000

        return numel

    def __getitem__(self, idx):
        windowSize=self.beforePts+self.afterPts+self.targetPts

        if self.limit:
            #uses a predefined list of data points
            fileIdx=self.dataDraws[idx,0]
            channelIdx=self.dataDraws[idx,1]
            randomIdx=self.dataDraws[idx,2]
            data,_ = self.raws[fileIdx][channelIdx,randomIdx:randomIdx+windowSize]
        else:
            #randomly selects a data point from all possible:
            fileIdx,randomChannelIdx,randomIdx,data=self.getAllowedDatapoint(returnData=True)
    
        #make sure there are no nan's in the data:
        assert(not np.any(np.isnan(data)))

        
        data = torch.tensor(data, dtype=torch.float32)

        if self.transform:
            data = self.transform(data)
        
        x12=(data[0,0:self.beforePts],data[0,-self.afterPts:])
        target=data[0,self.beforePts:(-self.afterPts)]

        return x12,target



class EEG_bids_dataset(EEG_dataset_flexible):
    def __init__(self, bidsPaths, beforePts,afterPts,targetPts, channelIdxs=1, transform=None,limit=None):
        self.transform = transform
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.channelIdxs=channelIdxs
        self.nChannels=len(channelIdxs) if isinstance(channelIdxs, (list,tuple,range)) else 1
        self.file_paths=[str(fp) for fp in bidsPaths]
        
        self.limit=limit

        self.preLoad()



def get_set_files(bidsDir,subjectIds,sessionIds,task):
    setPaths=[]
    for subId in subjectIds:
        for sessId in sessionIds:
            filePath=mb.BIDSPath(root=bidsDir,subject=subId,session=sessId,datatype='eeg',extension='.set',task=task)
            if os.path.exists(filePath):
                setPaths.append(filePath)
    return setPaths

if __name__ == "__main__":

    import mne_bids as mb
    bidsDir='C:\\Users\\au207178\\OneDrive - Aarhus universitet\\forskning\\EEGprediction\\\localData\\train\\\cleaned_1'

    subjectIds=mb.get_entity_vals(bidsDir,'subject',with_key=False)
    sessionIds=mb.get_entity_vals(bidsDir,'session')

    filePaths=get_set_files(bidsDir,subjectIds[0:5],sessionIds[0:3],'sleep')

    ds=EEG_bids_dataset(filePaths, beforePts=10,afterPts=10,targetPts=10, channelIdxs=[0,1],limit=10000)

    dl=torch.utils.data.DataLoader(ds, batch_size=1000)


    
    # import cProfile
    # cProfile.run('ds[0]')

    # cProfile.run('next(iter(dl))')

    # import time

    # start = time.perf_counter()
    ds[0][1:2]


    # end = time.perf_counter()
    # print(end - start)

    # start = time.perf_counter()
    # next(iter(dl))
    # end = time.perf_counter()
    # print(end - start)
        

#%%
