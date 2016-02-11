function y_out=low_filter(y)
%��������������ȡlinechart1�������κβ��������ȡlinechart2
n=length(y);
yfft=fftshift(fft(y));
yp=abs(yfft).^2;
figure,
subplot(121);
plot((1:n)-(n+1)/2,yp);
title('ԭʼ�ź�Ƶ��������');xlabel('Ƶ��');ylabel('����');
subplot(122);
plot(1:n,y);
title(strcat('ԭʼʱ���ź�'));xlabel('x');ylabel('y');

%% ��ͨ�˲�
threshold=n/16;
for i1=1:4
    yfft([1:floor((n+1)/2-threshold),ceil((n+1)/2+threshold):n],1)=0;
    yp=abs(yfft).^2;
    threshold=threshold/2;
    y2=real(ifft(ifftshift(yfft)));
    figure;
    subplot(121);
    plot((1:n)-(n+1)/2,yp);
    title(strcat('��ͨ�˲���',num2str(i1),'����ֵ������������'));xlabel('Ƶ��');ylabel('����');
    subplot(122);
    plot(1:n,y2);
    title(strcat('��ͨ�˲���',num2str(i1),'����ֵ������ʱ���ź�'));xlabel('x');ylabel('y');
end

%{
%% ��ֵб����б
s=median(abs((y2(2:end,1)-y2(1:end-1,1))./(y(2:end,1)-y(1:end-1,1))));
%��߱�
arfa1=s*(max(y)-min(y))/(max(y2)-min(y2));
figure;
subplot(211);
plot(y,y2);
set(gca,'DataAspectRatio',[arfa1 1 1]);
title('��ֵб����б');
xlabel('x');ylabel('y');
grid;
%% ƽ��б����б
s=mean(abs((y2(2:end,1)-y2(1:end-1,1))./(y(2:end,1)-y(1:end-1,1))));
%��߱�
arfa2=s*(max(y)-min(y))/(max(y2)-min(y2));
subplot(212);
plot(y,y2);
set(gca,'DataAspectRatio',[arfa2 1 1]);
title('ƽ��б����б');
xlabel('x');ylabel('y');
grid;
arfa=[arfa1,arfa2];
end
%}




