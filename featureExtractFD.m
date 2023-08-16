function [ff1] = featureExtractFD(data_H,data_F1,data_F2)
Fs = 2560;
T = 1/Fs;
L = 38400;             % Length of signal
t = (0:L-1)*T; 

f = Fs*(0:(L/2))/L;

fft_H=[];
for i=1:20
    Y_H=normalize(data_H);
    Y_H=fft(Y_H(:,i));
    P2=abs(Y_H/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_H=[fft_H P1];
end

fft_F1=[];
for i=1:20
    Y_F1=normalize(data_F1);
    Y_F1=fft(Y_F1(:,i));
    P2=abs(Y_F1/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_F1=[fft_F1 P1];
end

fft_F2=[];
for i=1:20
    Y_F2=normalize(data_F2);
    Y_F2=fft(Y_F2(:,i));
    P2=abs(Y_F2/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_F2=[fft_F2 P1];
end



fft_F1=fft_F1.';
fft_F2=fft_F2.';
fft_H=fft_H.';
figure;plot(f,fft_H(1,:));
hold on
plot(f,fft_F1(1,:));
plot(f,fft_F2(1,:));
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')
legend('Health','Faulty level 1','Faulty level 2')


data_fft=[fft_H; fft_F1; fft_F2];
n=20;
numerator=zeros(1,length(data_fft));
denominator=zeros(1,length(data_fft));
for k=1:length(data_fft)
    numerator(k)=numerator(k)+n*((mean(data_fft(1:20,k))-mean(data_fft(:,k)))^2);
    numerator(k)=numerator(k)+n*((mean(data_fft(21:40,k))-mean(data_fft(:,k)))^2);
    numerator(k)=numerator(k)+n*((mean(data_fft(41:60,k))-mean(data_fft(:,k)))^2);
    
    denominator(k)=denominator(k)+n*(var(data_fft(1:20,k)));
    denominator(k)=denominator(k)+n*(var(data_fft(21:40,k)));
    denominator(k)=denominator(k)+n*(var(data_fft(41:60,k)));
end
fisherS=numerator./denominator;

[fisherVal, loc]=sort(fisherS,'descend');
ff1=[data_fft(:,loc(1)) data_fft(:,loc(122))];
end

