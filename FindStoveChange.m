function [indexChange,sHWP,dHWP]=FindStoveChange(hotWindPress)
md=zeros(length(hotWindPress),1);
sd=zeros(length(hotWindPress),1);
for i1=1:length(hotWindPress)
    index=(max(1,i1-720):i1);
    tempData=hotWindPress(index);
    md(i1)=median(tempData);
    sd(i1)=std(tempData);
end
sHWP=(hotWindPress-md)./max(sd,0.0001);
dHWP=hotWindPress(2:end,:)-hotWindPress(1:end-1,:);
dHWP=[0;dHWP/std(dHWP)];

avgLength=338;
lastChange=0;
width=60;
indexChange=false(length(hotWindPress),1);
while lastChange<length(hotWindPress)-avgLength
    len0=lastChange;
    lastChange=lastChange+avgLength;%下一个可能换炉的地方
    search=[lastChange-floor(avgLength/2),min(lastChange+floor(avgLength/2),length(hotWindPress)-1)];%附近的可能区域
    least=find(dHWP(search(1):search(2))<-2.5,1,'first');
    if isempty(least)
        continue;
    end
    least=search(1)+least-1;
    search(1)=least;
    search(2)=min(search(1)+width,length(hotWindPress)-1);
    [~,highest]=max(dHWP(search(1):search(2)));
    highest=highest+least-1;
    if dHWP(highest)<1.9
        lastChange=lastChange-floor(avgLength/2);
        continue;
    end
    [~,lastChange]=min(hotWindPress(least:highest));%找到附近最小值
    lastChange=lastChange+least-1;
    avgLength=ceil(0.9*avgLength+0.1*(lastChange-len0));
%     重新调整least highest
    least=min(least,lastChange-10);
    highest=max(highest,lastChange+10);
    
    doulbt1=least:-1:max(1,least-ceil(width/2-(lastChange-least)));
    index1=find(hotWindPress(doulbt1)>hotWindPress(doulbt1-1),1,'first');
    if isempty(index1)
        index1=length(doulbt1)+1;
    end
    doulbt2=highest:min(highest+ceil(width/2-(highest-lastChange)),length(hotWindPress)-1);
    index2=find(hotWindPress(doulbt2)>hotWindPress(doulbt2+1),1,'first');
    if isempty(index2)
        index2=length(doulbt2)+1;
    end
    indexChange(least-index1+1:highest+index2-1,1)=true;
%     range=lastChange-index1+1:lastChange+index2-1;%[doulbt1(end:-1:1),doulbt2];
%     plot(1:length(hotWindPress),hotWindPress,range,hotWindPress(range))
end
end