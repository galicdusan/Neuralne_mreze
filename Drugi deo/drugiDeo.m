clear all
close all
clc

%ucitavanje podataka
load connect-4.mat -ASCII
connect_4 = connect_4';
X = connect_4(1:42, :);
Y = connect_4(43:45, :);
Y_win = connect_4(43, :);
Y_loss = connect_4(44, :);
Y_draw = connect_4(45, :);
X_win = X(:, Y_win==1);
X_loss = X(:, Y_loss==1);
X_draw = X(:, Y_draw==1);

%razdvajanje na trening i test skupove

%X_win
ind = randperm(length(X_win));
br = round(0.9*length(ind));

X_win_train = X_win(:, ind(1:br));
Y_win_train = zeros(3, br);
Y_win_train(1,:) = ones(1, br);

X_win_test = X_win(:, ind(br+1:end));
Y_win_test = zeros(3, length(br+1:length(ind)));
Y_win_test(1,:) = ones(1, length(br+1:length(ind)));

%X_loss
ind = randperm(length(X_loss));
br = round(0.9*length(ind));

X_loss_train = X_loss(:, ind(1:br));
Y_loss_train = zeros(3, br);
Y_loss_train(2,:) = ones(1, br);

X_loss_test = X_loss(:, ind(br+1:end));
Y_loss_test = zeros(3, length(br+1:length(ind)));
Y_loss_test(2,:) = ones(1, length(br+1:length(ind)));

%X_draw
ind = randperm(length(X_draw));
br = round(0.9*length(ind));

X_draw_train = X_draw(:, ind(1:br));
Y_draw_train = zeros(3, br);
Y_draw_train(3,:) = ones(1, br);

X_draw_test = X_draw(:, ind(br+1:end));
Y_draw_test = zeros(3, length(br+1:length(ind)));
Y_draw_test(3,:) = ones(1, length(br+1:length(ind)));

%spajamo
X_train = [X_win_train, X_loss_train, X_draw_train];
Y_train = [Y_win_train, Y_loss_train, Y_draw_train];
ind = randperm(length(X_train));
X_train = X_train(:, ind);
Y_train = Y_train(:, ind);

X_test = [X_win_test, X_loss_test, X_draw_test];
Y_test = [Y_win_test, Y_loss_test, Y_draw_test];
ind = randperm(length(X_test));
X_test = X_test(:, ind);
Y_test = Y_test(:, ind);

%strukture za parametre
structure1 = {[15 15 15], [15 15], [10 10 10], [30 30], [20 20], [25 25], [20 20 20], [30 30 30], [8 8]}; % 1, [1 1], 2, [2 2], 3, [3 3], 5, [5 5] , [8 8]
structure2 = {'tansig', 'logsig'}; % 'tansig', 'logsig', 'purelin'
structure3 = {0.004, 0.006}; % 0.1, 0.05, 0.005, 0.01,  ///  0.004, 0.006
structure4 = {0.9, 0.8}; % 0.9, 0.8, 0.7
structure5 = {15, 6, 25}; % 37, 25, 50  ///  6, 15 25
structure6 = {600, 300, 400}; % 500, 800, 1200, /// 400, 300, 600
structure7 = {0.3, 0.2}; %0.85, 0.9, 0.8 ///  0.2, 0.3
structure8 = {0.005, 0.001}; % learning rate /// 0.005, 0.001

%inicijalizujemo najbolje vrednosti
najSloj = structure1{1};
najFunk = structure2{1};
najReg = structure3{1};
najRatio = structure4{1};
najFail = structure5{1};
najEpohe = structure6{1};
najTezina = structure7{1};
najLr = structure8{1};
najFscore = 0;
najAcc = 0;

%vrtimo hiperparametre
for i1 = 1:length(structure1)
    for i2 = 1:length(structure2)
        for i3 = 1:length(structure3)
            for i4 = 1:length(structure4)
                for i5 = 1:length(structure5)
                    for i6 = 1:length(structure6)
                        for i7 = 1:length(structure7)
                            for i8 = 1:length(structure8)
                        
                                net = patternnet(structure1{i1});

                                net.divideFcn = 'dividerand';
                                net.trainParam.goal = 1e-6;
                                net.trainParam.min_grad = 0;
                                net.trainparam.showWindow = false;

                                for k = 1:length(structure1{i1})
                                    net.layers{k}.transferFcn = structure2{i2};
                                end

                                net.performParam.regularization = structure3{i3};
                                net.divideParam.trainRatio = structure4{i4};
                                net.divideParam.valRatio = 1 - structure4{i4};
                                net.divideParam.testRatio = 0;
                                net.trainParam.max_fail = structure5{i5};
                                net.trainParam.epochs = structure6{i6};

                                we = (Y_train(1,:)== 1)*structure7{i7} + (Y_train(3,:)== 1) + (Y_train(2,:)== 1)*structure7{i7}*2;

                                net.trainParam.lr = structure8{i8};

                                [net,tr] = train(net,X_train,Y_train, {}, {}, we);
                %                [net,tr] = train(net, X_train,Y_train);
                                valInd = tr.valInd;

                                X_val = X_train(:, valInd);
                                Y_val = Y_train(:, valInd);
                                Y_val_pred = sim (net, X_val);

                                [c,cm,ind,per] = confusion(Y_val, Y_val_pred);
                                cm = cm';
                                precision = cm(1,1)/( cm(1,1)+cm(1,2)+cm(1,3) );
                                recall = cm(1,1)/( cm(1,1)+cm(2,1)+cm(3,1) );
                                fscore = 2 * (precision * recall) / (precision + recall);
                                accuracy = ( cm(1,1)+cm(2,2)+cm(3,3) )/( cm(1,1)+cm(2,2)+cm(1,2)+cm(2,1)+cm(3,3)+cm(3,1)+cm(3,2)+cm(1,3)+cm(2,3) );
                                if(fscore > najFscore)
                                    najFscore = fscore;
                                    najSloj = structure1{i1};
                                    najFunk = structure2{i2};
                                    najReg = structure3{i3};
                                    najRatio = structure4{i4};
                                    najFail = structure5{i5};
                                    najEpohe = structure6{i6};
                                    najTezina = structure7{i7};
                                    najLr = structure8{i8};
                                    najAcc = accuracy;
                                end
                                %figure(4)
                                %hold all
                                %plotconfusion(Y_val, Y_val_pred)
                            end
                        end
                    end
                end
            end
        end
    end
end
disp(' ')
disp('Rezultat krosvalidacije: ')
disp(['Najbolja struktura:  ' , num2str(najSloj)])
disp(['Najbolja aktivaciona funkcija:  ' , char(najFunk)])
disp(['Najbolja podela trening skupa:  ' , num2str(najRatio)])
disp(['Najbolji koeficijent regularizacije:  ' , num2str(najReg)])
disp(['Najbolja duzina treniranja sa ovim parametrima:  ' , num2str(najEpohe)])
disp(['Najbolja duzina epoha za validaciju  ' , num2str(najFail)])
disp(['Najbolja greska  ' , num2str(najTezina)])
disp(' ')
disp(['Najbolji accuracy:  ' , num2str(najAcc)])
disp(['Najbolji f1:  ' , num2str(najFscore)])
disp(' ')
disp(' ')

%treniramo sa najboljim hiperparametrima
net = patternnet (najSloj);
for k = 1:length(najSloj)
    net.layers{k}.transferFcn = najFunk;
end
net.performParam.regularization = najReg;
net.divideParam.trainRatio = najRatio;
net.divideParam.valRatio = 1 - najRatio;
net.divideParam.testRatio = 0;
net.trainParam.max_fail = najFail;
net.trainParam.epochs = najEpohe;
we = (Y_train(1,:)== 1)*najTezina + (Y_train(3,:)== 1) + (Y_train(2,:)== 1)*najTezina*2;
net.trainParam.lr = najLr;
[net, tr] = train(net, X_train, Y_train, {}, {}, we);

%pokazujemo krivu obucavanja za finalnu mrezu
figure(5)
hold all
plotperform(tr)

%treniranje NM sa dobijenim najboljim hiperparametrima
net.trainParam.showWindow = true;
net.divideParam.trainRatio = 1;
net.divideParam.valRatio = 0;
[net, tr] = train(net, X_train, Y_train, {}, {}, we);

%kao ulaz ubacujemo trening i test skupove u istreniranu mrezu
Y_test_pred = sim(net, X_test);
Y_train_pred = sim(net, X_train);

%crtamo konfuzionu za trening, racunamo mere preciznosti
figure(6)
hold all
title('Train set')
plotconfusion(Y_train, Y_train_pred)

[c,cm,ind,per] = confusion(Y_train, Y_train_pred);
cm = cm';
precision_train = cm(1,1)/( cm(1,1)+cm(1,2)+cm(1,3) );
recall_train = cm(1,1)/( cm(1,1)+cm(2,1)+cm(3,1) );
fscore_train = 2 * (precision_train * recall_train) / (precision_train + recall_train);
accuracy_train = ( cm(1,1)+cm(2,2)+cm(3,3) )/( cm(1,1)+cm(2,2)+cm(1,2)+cm(2,1)+cm(3,3)+cm(3,1)+cm(3,2)+cm(1,3)+cm(2,3) );

%crtamo konfuzionu za test, racunamo mere preciznosti
figure(7)
hold all
title('Test set')
plotconfusion(Y_test, Y_test_pred)

[c,cm,ind,per] = confusion(Y_test, Y_test_pred);
cm = cm';
precision_test = cm(1,1)/( cm(1,1)+cm(1,2)+cm(1,3) );
recall_test = cm(1,1)/( cm(1,1)+cm(2,1)+cm(3,1) );
fscore_test = 2 * (precision_test * recall_test) / (precision_test + recall_test);
accuracy_test = ( cm(1,1)+cm(2,2)+cm(3,3) )/( cm(1,1)+cm(2,2)+cm(1,2)+cm(2,1)+cm(3,3)+cm(3,1)+cm(3,2)+cm(1,3)+cm(2,3) );

disp(' ')
disp('Trenirajuci skup: ')
disp(['Accuracy:  ' , num2str(accuracy_train)])
disp(['f1:  ' , num2str(fscore_train)])
disp(['precision:  ' , num2str(precision_train)])
disp(['recall:  ' , num2str(recall_train)])
disp(' ')

disp(' ')
disp('Testirajuci skup: ')
disp(['Accuracy:  ' , num2str(accuracy_test)])
disp(['f1:  ' , num2str(fscore_test)])
disp(['precision:  ' , num2str(precision_test)])
disp(['recall:  ' , num2str(recall_test)])
disp(' ')


