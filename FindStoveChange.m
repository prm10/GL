function [indexChange]=FindStoveChange(hotWindPress)
md=zeros(length(hotWindPress),1);
sd=zeros(length(hotWindPress),1);
for i1=1:length(hotWindPress)
    index=(max(1,i1-720):i1);
    tempData=hotWindPress(index);
    md(i1)=median(tempData);
    sd(i1)=std(tempData);
end
flag=(hotWindPress-md);
avgLength=338;
lastChange=0;
width=100;
indexChange=false(length(hotWindPress),1);
while lastChange<length(hotWindPress)-avgLength
    len0=lastChange;
    lastChange=lastChange+avgLength;
    search=[lastChange-floor(avgLength/2),min(lastChange+floor(avgLength/3),length(hotWindPress)-1)];
    [~,lastChange]=min(flag(search(1):search(2)));
    lastChange=lastChange+search(1)-1;
    avgLength=ceil(0.9*avgLength+0.1*(lastChange-len0));
    doulbt1=lastChange:-1:max(1,lastChange-width);
    index1=find(hotWindPress(doulbt1)>hotWindPress(doulbt1-1),1,'first');
    doulbt2=lastChange:min(lastChange+width,length(hotWindPress)-1);
    index2=find(hotWindPress(doulbt2)>hotWindPress(doulbt2+1),1,'first');
    indexChange(lastChange-index1:lastChange+index2,1)=true;
%     range=lastChange-index1+1:lastChange+index2-1;%[doulbt1(end:-1:1),doulbt2];
%     plot(1:length(hotWindPress),hotWindPress,range,hotWindPress(range))
end
end