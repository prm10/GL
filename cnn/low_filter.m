function y_out=low_filter(y)
%不输入参数，则读取linechart1，输入任何参数，则读取linechart2
n=length(y);
yfft=fftshift(fft(y));
yp=abs(yfft).^2;
figure,
subplot(121);
plot((1:n)-(n+1)/2,yp);
title('原始信号频域能量谱');xlabel('频率');ylabel('能量');
subplot(122);
plot(1:n,y);
title(strcat('原始时域信号'));xlabel('x');ylabel('y');

%% 低通滤波
threshold=n/16;
for i1=1:4
    yfft([1:floor((n+1)/2-threshold),ceil((n+1)/2+threshold):n],1)=0;
    yp=abs(yfft).^2;
    threshold=threshold/2;
    y2=real(ifft(ifftshift(yfft)));
    figure;
    subplot(121);
    plot((1:n)-(n+1)/2,yp);
    title(strcat('低通滤波第',num2str(i1),'次阈值迭代后能量谱'));xlabel('频率');ylabel('能量');
    subplot(122);
    plot(1:n,y2);
    title(strcat('低通滤波第',num2str(i1),'次阈值迭代后时域信号'));xlabel('x');ylabel('y');
end

%{
%% 中值斜率倾斜
s=median(abs((y2(2:end,1)-y2(1:end-1,1))./(y(2:end,1)-y(1:end-1,1))));
%宽高比
arfa1=s*(max(y)-min(y))/(max(y2)-min(y2));
figure;
subplot(211);
plot(y,y2);
set(gca,'DataAspectRatio',[arfa1 1 1]);
title('中值斜率倾斜');
xlabel('x');ylabel('y');
grid;
%% 平均斜率倾斜
s=mean(abs((y2(2:end,1)-y2(1:end-1,1))./(y(2:end,1)-y(1:end-1,1))));
%宽高比
arfa2=s*(max(y)-min(y))/(max(y2)-min(y2));
subplot(212);
plot(y,y2);
set(gca,'DataAspectRatio',[arfa2 1 1]);
title('平均斜率倾斜');
xlabel('x');ylabel('y');
grid;
arfa=[arfa1,arfa2];
end
%}




