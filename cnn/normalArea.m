function index=normalArea(ns,width)
% s=find(~ns);% abnormal state
s=find(smooth(double(ns),360)<0.7);
index=[];
len=0;
for i1=1:length(s)-1
    if s(i1+1)-s(i1)>=width
        index=[index;[s(i1)+360,s(i1+1)-360]];
        len=len+s(i1+1)-s(i1);
    end
end
disp(strcat('normal rate: ',num2str(len/length(ns))));