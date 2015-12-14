function [input,decode,predict]=GenerateData(data,lengthD,lengthP,indexD,indexP)
input=cell(0);
decode=cell(0);
predict=cell(0);
%从至少两倍lengthD的地方开始生成样本
lengthV=size(data,1)-2*lengthD-lengthP;%valid length
for i1=1:lengthV
    n=i1+2*lengthD;
    input=[input;data(1:n,:)];
    decode=[decode;data(n:-1:n-lengthD+1,indexD)];
    predict=[predict;data(n+1:n+lengthP,indexP)];
end



