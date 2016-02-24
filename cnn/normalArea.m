function [index,ignore]=normalArea(ns,width)
% s=find(~ns);% abnormal state
ns1=smooth(double(ns),360);
ns2=smooth(double(ns),60);
s=find(ns1<0.7);
index=[];
ignore=cell(0);
len=0;
for i1=1:length(s)-1
    if s(i1+1)-s(i1)>=width %normal state for a long time
        index=[index;[s(i1)+360,s(i1+1)-360]];
        ignore=[ignore;find(ns2(s(i1)+360:s(i1+1)-360)<0.5)];
        len=len+s(i1+1)-s(i1);
    end
end
disp(strcat('normal rate: ',num2str(len/length(ns))));