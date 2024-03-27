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
        self.channelIdxs=channelIdxs
        self.nChannels=len(channelIdxs) if isinstance(channelIdxs, (list,tuple,range)) else 1
        self.file_paths=[str(fp) for fp in bidsPaths]
        self.limit=limit #if 
        self.train_size = train_size

        maxFilesLoaded=self.determineMemoryCapacity()

        #preload:
        self.raws=[]
        nfilesToLoad=min(maxFilesLoaded,len(self.file_paths))
        # Take nfiles randomly but not replaced so this is just all the files ??? 
        # Is it to get them in random order? THIS LINE BELOW DOES NOTHING 
        fileIdxToLoad=np.random.choice(len(self.file_paths),nfilesToLoad,replace=False)

        for fileIdx in fileIdxToLoad:
            try:
                print('*'*65)
                print(self.file_paths[fileIdx])
                tempRaw=mne.io.read_raw_eeglab(self.file_paths[fileIdx],preload=True,verbose=False)

                channelsToExclude=(1- np.isin(range(0,tempRaw.info['nchan']),self.channelIdxs)).nonzero()[0].astype('int')
                # import pdb; pdb.set_trace()
                channelsToExclude=np.asarray(tempRaw.ch_names)[channelsToExclude]
                tempRaw.drop_channels(channelsToExclude)

                if self.transform:
                    tempRaw = self.transform(tempRaw)

                self.raws.append(tempRaw)
            except Exception as error:
                print(error)
                print(self.file_paths[fileIdx])

            # return tempRaw
        
        # #loading raw files in parallel:
        # print('Loading files in parallel:')
        # self.raws = Parallel(n_jobs=np.min((3,cpu_count())), verbose=1, backend="threading")(map(delayed(lambda fileIdx: rawLoader(self,fileIdx)),fileIdxToLoad))
        
        
        if limit:
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

    def getAllowedDatapoint(self, returnData=False):
        windowSize=self.beforePts+self.afterPts+self.targetPts
        #keep looking until we find a data window without nan's
        data=np.nan

        while np.any(np.isnan(data)):            
            randFileIdx=np.random.randint(0, len(self.raws))    
            randomChannelIdx=np.random.choice(self.nChannels)
            randomIdx=np.random.randint(0, self.raws[randFileIdx].n_times-windowSize)

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
            numel=self.train_size
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

        
        x12 = (data[0,0:self.beforePts], data[0,-self.afterPts:])
        target = data[0,self.beforePts:(-self.afterPts)]


        return x12,target



# if __name__ == "__main__":
    # import mne_bids as mb

    # bidsPath="C:/Users/au207178/OneDrive - Aarhus Universitet/forskning/EEGprediction/localData/train/"


    # subjectIds=mb.get_entity_vals(bidsPath,'subject',with_key=False)
    # trainIds=subjectIds.copy()
    # trainIds.pop(1)
    # trainPaths=returnFilePaths(bidsPath,trainIds,sessionIds=['001', '002', '003', '004'])

    # def myFun(hep,fpath):
    #     try:
    #         tempRaw=mne.io.read_raw_eeglab(fpath,preload=True,verbose=False)
    #         return tempRaw
    #     except Exception as error:
    #         print(error)
    #         print(fpath)

    # myFun2 = lambda fpath: myFun(1,fpath)

    # raws = Parallel(n_jobs=2, verbose=1, backend="threading")(map(delayed(lambda fpath: myFun(1,fpath)), trainPaths))


    # def mytransform(raw):
    #     raw.filter(0.1,40)
    #     raw._data=raw._data*1e6
    #     return raw

    # bidsPath="C:/Users/au207178/OneDrive - Aarhus Universitet/forskning/EEGprediction/localData/train/"


    # subjectIds=mb.get_entity_vals(bidsPath,'subject',with_key=False)
    # trainIds=subjectIds.copy()
    # trainIds.pop(1)
    # trainPaths=returnFilePaths(bidsPath,trainIds,sessionIds=['001', '002', '003', '004'])

    # ds_train=EEG_dataset_from_paths(trainPaths, beforePts=500,afterPts=500,targetPts=100, channelIdxs=[1,7,23],preprocess=False,limit=None,transform=mytransform)




        

