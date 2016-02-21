function data2csv(data_seg,len_data,len_target,csvPath,csvName)
% 以后考虑下对目标信号进行滤波
% low_filter(data1(range,ipt(4)))
%% get data and target
% sv_seg=sv(range,:);% stove change records of concerned data
len_valid=size(data_seg,1)-len_data-len_target+1;
target_std=zeros(len_valid,size(data_seg,2));
target_mean=zeros(len_valid,size(data_seg,2));
for i1=1:size(target_std,1)
    data_target=data_seg(i1+len_data:i1+len_data+len_target-1,:);
    target_mean(i1,:)=median(data_target);
    target_std(i1,:)=std(data_target,0,1);
end
data=data_seg(1:end-len_target,:);
csvwrite(strcat(csvPath,'data_',csvName,'.csv'),data);
csvwrite(strcat(csvPath,'target_mean_',csvName,'.csv'),target_mean);
csvwrite(strcat(csvPath,'target_std_',csvName,'.csv'),target_std);