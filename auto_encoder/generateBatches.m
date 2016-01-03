function dataOut=generateBatches(train_data,batchSize,timeStep,timeLen)
batches=floor((size(train_data,1)-timeLen)/batchSize/timeStep);
dataOut=zeros(batchSize,timeLen,batches);
istart=0;
ibatches=1;
ibatchSize=1;
while ibatches<=batches
    dataOut(ibatchSize,:,ibatches)=train_data(istart+1:istart+timeLen,1);
    istart=istart+timeStep;
    if ibatchSize==batchSize
        ibatchSize=1;
        ibatches=ibatches+1;
    else
        ibatchSize=ibatchSize+1;
    end
end
