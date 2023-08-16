clc; clear ;close all;
%%  Training set
address='C:\Users\LORSMIP\Desktop\homework3\Training\Healthy\';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_H=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_H=[data_H cube.data];
end
[mean_H,var_H,skew_H,kurts_H,rms_H,pp_H,median_H,energy_H,Entropy_H] = featureExtractTD(data_H);
feature_H=[mean_H',var_H',skew_H',kurts_H',rms_H',pp_H',median_H',energy_H',Entropy_H'];


address='C:\Users\LORSMIP\Desktop\homework3\Training\Faulty\unbalance 1\';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_F1=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_F1=[data_F1 cube.data];
end
[mean_F1,var_F1,skew_F1,kurts_F1,rms_F1,pp_F1,median_F1,energy_F1,Entropy_F1] = featureExtractTD(data_F1);
feature_F1=[mean_F1',var_F1',skew_F1',kurts_F1',rms_F1',pp_F1',median_F1',energy_F1',Entropy_F1'];



address='C:\Users\LORSMIP\Desktop\homework3\Training\Faulty\unbalance 2\';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_F2=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_F2=[data_F2 cube.data];
end
[mean_F2,var_F2,skew_F2,kurts_F2,rms_F2,pp_F2,median_F2,energy_F2,Entropy_F2] = featureExtractTD(data_F2);
feature_F2=[mean_F2',var_F2',skew_F2',kurts_F2',rms_F2',pp_F2',median_F2',energy_F2',Entropy_F2'];

frequency=featureExtractFD(data_H,data_F1,data_F2);

feature_H=[frequency(1:20,:) feature_H ];
feature_F1=[frequency(21:40,:) feature_F1 ];
feature_F2=[frequency(41:60,:) feature_F2 ];

data_feature=[feature_H;feature_F1;feature_F2];


n=20
[row,col]=size(data_feature);
numerator=zeros(1,col);
denominator=zeros(1,col);
%data_feature=normalize(data_feature);
for k=1:col
    numerator(k)=n*((mean(data_feature(1:20,k))-mean(data_feature(:,k)))^2)...
    +n*((mean(data_feature(21:40,k))-mean(data_feature(:,k)))^2)...
    +n*((mean(data_feature(41:60,k))-mean(data_feature(:,k)))^2);
    
    denominator(k)=n*(var(data_feature(1:20,k)))+n*(var(data_feature(21:40,k)))+n*(var(data_feature(41:60,k)));
end
fisherS=numerator./denominator;

c = categorical({'20.8 Hz','84.6 Hz','Mean','Variance','Skewness','Kurtosis','RMS','Peak-to-peak','Median','Energy','Entropy'});
figure
bar(c,fisherS);
title('Fisher score')
ylabel('score')
xlabel('Features')

csvwrite('data_feature.csv',[data_feature(:,1:2) [zeros(20,1);ones(20,1);ones(20,1)+1]]);

figure
hist1=histogram(data_feature(1:20,1));
hold on
hist2=histogram(data_feature(21:40,1));
hist3=histogram(data_feature(41:60,1));
xlabel('value')
ylabel('# observation')
title('20.8 Hz-Histogram')
legend('healthy','unbalance 1','unbalance 2')


figure
hist1=histogram(data_feature(1:20,2));
hold on
hist2=histogram(data_feature(21:40,2));
hist3=histogram(data_feature(41:60,2));
xlabel('value')
ylabel('# observation')
title('84.6 Hz-Histogram')
legend('healthy','unbalance 1','unbalance 2')



%% Testing set
address='C:\Users\LORSMIP\Desktop\homework3\Testing\';
temp=dir([address,'*.txt']);
temp_size=size(temp);
data_T=[];
for q=1:temp_size(1)
    cube = importdata([address,temp(q).name]);
    data_T=[data_T cube.data];
end

Fs = 2560;
T = 1/Fs;
L = 38400;             % Length of signal
t = (0:L-1)*T; 

f = Fs*(0:(L/2))/L;

fft_T=[];
for i=1:30
    Y_T=normalize(data_T);
    Y_T=fft(Y_T(:,i));
    P2=abs(Y_T/L);
    P1=P2(1:L/2+1);
    P1(2:end-1)=2*P1(2:end-1);
    fft_T=[fft_T P1];
end



fft_T=fft_T.';
data_test=[fft_T(:,313) fft_T(:,1270) [zeros(10,1);ones(10,1);ones(10,1)+1]];
csvwrite('data_testing.csv',data_test);



 
load data_feature.csv
[beta,dev,stats] = mnrfit(data_feature(:,1:2),categorical(data_feature(:,3)));
pihat = mnrval(beta,data_feature(:,1:2));
[~,loc]=max(pihat');
loc = [loc-1]'; 
load data_testing.csv
pihat_test = mnrval(beta,data_testing(:,1:2));
[~,loc_test]=max(pihat_test');
loc_test = [loc_test-1]'; 

loc_test=categorical(loc_test);
target=categorical(data_testing(:,3));

figure;plotconfusion(target,loc_test);
title('logistic regression - confusion matrix')







