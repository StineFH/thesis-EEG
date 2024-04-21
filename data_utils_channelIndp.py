import torch
import os
# import re
import mne
import numpy as np
from joblib import Parallel, delayed, cpu_count
import mne_bids as mb


def returnFilePaths(bidsDir,subjectIds=None,sessionIds=None,taskIds=None):
    """
    wrapper for get_entity_vals and BIDSPath, to get all files matching certain ID's
    """
    def debugMessage(input,inputName):
        if type(input) is not list:
            raise Exception( "returnFilepaths expects a list or None for " + inputName + " Id's. Consider enclosing id in '[]'" )

    #list of subjects:
    if not subjectIds:
        subjectIds=mb.get_entity_vals(bidsDir,'subject')
        if len(subjectIds)==0:
            subjectIds=[None]
    debugMessage(subjectIds,'subject')

    #list of sessions:
    if not sessionIds:
        sessionIds=mb.get_entity_vals(bidsDir,'session')
        if len(sessionIds)==0:
            sessionIds=[None]
    debugMessage(sessionIds,'session')

    #list of tasks:
    if not taskIds:
        taskIds=mb.get_entity_vals(bidsDir,'task')
        if len(taskIds)==0:
            taskIds=[None]
    debugMessage(taskIds,'task')

    print('Subject Ids:',subjectIds)
    print('Session Ids:',sessionIds)
    print('Task ids:',taskIds)

    #and here we just check and add all possible combinations:
    filePaths=[]
    for sub in subjectIds:
        for ses in sessionIds:
            for task in taskIds:
                try:
                    temp=mb.BIDSPath(root=bidsDir,subject=sub,session=ses,task=task,datatype='eeg',extension='.set',check=False)
                    if os.path.isfile(str(temp)):
                        filePaths.append(str(temp))
                except Exception as error:
                    print(error)
                    print(sub,ses,task)

    return filePaths


class EEG_dataset_from_paths(torch.utils.data.Dataset):
    def __init__(self, bidsPaths, beforePts, afterPts, targetPts, channelIdxs=1, 
                 transform=None,preprocess=False,limit=None, train_size = 100000):
        self.transform = transform
        self.beforePts = beforePts
        self.afterPts = afterPts
        self.targetPts = targetPts
        self.channelIdxs= channelIdxs if isinstance(channelIdxs, list) else [channelIdxs]
        self.nChannels=len(channelIdxs) if isinstance(channelIdxs, (list,tuple,range)) else 1
        self.file_paths=[str(fp) for fp in bidsPaths]
        self.limit=limit #if 
        self.train_size = train_size

        maxFilesLoaded=self.determineMemoryCapacity()

        #preload:
        self.raws=[]
        nfilesToLoad=min(maxFilesLoaded,len(self.file_paths))
        fileIdxToLoad=np.random.choice(len(self.file_paths),nfilesToLoad,replace=False)

        for fileIdx in fileIdxToLoad:
            try:
                print('*'*65)
                print(self.file_paths[fileIdx])
                tempRaw=mne.io.read_raw_eeglab(self.file_paths[fileIdx],preload=True,verbose=False)

                channelsToExclude=(1- np.isin(range(0,tempRaw.info['nchan']),self.channelIdxs)).nonzero()[0].astype('int')
                channelsToExclude=np.asarray(tempRaw.ch_names)[channelsToExclude]
                tempRaw.drop_channels(channelsToExclude)

                if self.transform:
                    tempRaw = self.transform(tempRaw)

                self.raws.append(tempRaw)
            except Exception as error:
                print(error)
                print(self.file_paths[fileIdx])
        
        if limit:
            self.dataDraws=np.zeros((self.__len__(),2),np.int64) #columns for: file, time
            print('Preparing ready-made data draws...')

            def myfun(arg):
                result=self.getAllowedDatapoint()
                return result

            results = Parallel(n_jobs=np.max((1,cpu_count()-1)), verbose=1, backend="threading")(map(delayed(myfun), range(self.__len__())))

            self.dataDraws=np.asarray(results)

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

    def getAllowedDatapoint(self, returnData=False):
        windowSize=self.beforePts+self.afterPts+self.targetPts
        #keep looking until we find a data window without nan's
        data=np.nan

        while np.any(np.isnan(data)):            
            randFileIdx=np.random.randint(0, len(self.raws))    
            randomIdx=np.random.randint(0, self.raws[randFileIdx].n_times-windowSize)

            data=[]
            for ch in range(0, self.nChannels):
                data_i,_=self.raws[randFileIdx][ch,randomIdx:randomIdx+windowSize]
                data.append(data_i)
            data = np.vstack(data).reshape(1,self.nChannels, windowSize)
        
        if returnData:
            return randFileIdx,randomIdx,data
        else:
            return randFileIdx,randomIdx
        

    def __len__(self):
        if self.limit:
            numel=self.limit
        else:
            # numel=int(np.sum([raw.n_times for raw in self.raws])*self.nChannels/2)
            numel=self.train_size
        return numel

    def __getitem__(self, idx):
        windowSize=self.beforePts+self.afterPts+self.targetPts

        if self.limit:
            #uses a predefined list of data points
            fileIdx=self.dataDraws[idx,0]
            randomIdx=self.dataDraws[idx,1]
            data=[]
            for ch in range(0, self.nChannels):
                data_i,_=self.raws[fileIdx][ch,randomIdx:randomIdx+windowSize]
                data.append(data_i)
            data = np.vstack(data).reshape(1,self.nChannels, windowSize)
        else:
            #randomly selects a data point from all possible:
            fileIdx,randomIdx,data=self.getAllowedDatapoint(returnData=True)
    
        #make sure there are no nan's in the data:
        assert(not np.any(np.isnan(data)))

        
        data = torch.tensor(data, dtype=torch.float32)

        if self.afterPts == 0:
            x12 = (data[0, :, 0:self.beforePts])
            target = data[0, :, self.beforePts:]
        else: 
            x12 = (data[0, :,0:self.beforePts], data[0, :,-self.afterPts:])
            target = data[0, :,self.beforePts:(-self.afterPts)]

        return x12,target

