function index=normalArea(ns,width)
T=size(ns);
s=find(~ns);
index=[];
for i1=1:length(s)-1
    if s(i1+1)-s(i1)>=width
        index=[index;[s(i1)+1,s(i1+1)-360]];
    end
end