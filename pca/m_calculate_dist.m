clc;close all;clear;
range=5;
n=1000;
d=zeros(n,1);
for i1=1:n
    value=range*i1/n;
    d(i1,1)=2*(1-integral(@(x) exp(-x.^2/2),-100,value)/sqrt(2*pi));
end
loc=0:range/n:range;
plot(loc(2:end),d);
save('dist_map.mat','range','n','d');