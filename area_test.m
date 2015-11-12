close all;clear;clc;
% n：每行小圆个数
% r1：小圆半径
% r2：大圆半径
% o2：大圆圆心位置
% flag：0代表稀疏排列，1代表紧致排列
n=10;
r1=2;
r2=12;
o2=[27,15];
flag=1;
area1=area_in_a_big_circle(n,r1,r2,o2,flag);
flag=0;
area2=area_in_a_big_circle(n,r1,r2,o2,flag);