% NN Multilayer Perceptron

clear all
close all
clc

for Sujetos = 1:1
    
    clearvars -except Metrics Sujetos

    % Load Data
    load(['..\Features\PSD',num2str(Sujetos),'.mat'])
    
    for Runs1 = 1:5

        % Define features and classes matrix to train 
        Features = [DataPSD(Runs1).cross; DataPSD(Runs1).right];
        Class = [ones(size(DataPSD(Runs1).cross,1),1)*1; ones(size(DataPSD(Runs1).right,1),1)*2];
        Label = Class;
    
        % Cross Validation
        hpartition = cvpartition(length(Features),'kFold',5);
    
        % CONFIGURATION AND TRAINING OF NEURAL NETWORK
        for Hidd = 1:10
            Acc_base = 0;
            for Inic = 1:100
                Start_opt1 = tic; 
                for kfold = 1:5
                    
                    % Data
                    xtrain = Features(~hpartition.test(kfold),:);
                    ytrain = Label(~hpartition.test(kfold),:);
                    xtest = Features(hpartition.test(kfold),:);
                    ytest = Label(hpartition.test(kfold),:);
                    
                    % Adjust data
                    xtest = (xtest - mean(xtest)) ./ std(xtest); % Normalized Z-score
                    xtrain = (xtrain - mean(xtrain)) ./ std(xtrain); % Normalized Z-score
                    
                    % Parameters
                    Start_opt2 = tic;
                    input_layer_size  = size(xtrain,2); % Input layer size correspond to the size of the used features 
                    hidden_layer_size = Hidd; % Hidden layer can change, but in this ocassion have a value of 10 
                    num_labels = max(ytrain); % The output layer correspond to the predicted class, for this case are 6 classes
                    
                    % Inizializing parameters
                    initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size); % Theta 1 and Theta 2 are initialized of random form between values of -epsilon/2 and epsilon/2
                    initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
            
                    initial_nn_params = [initial_Theta1(:); initial_Theta2(:)];
            
                    options = optimset('MaxIter', 50); % The maximum number of iterations is established at 50
            
                    lambda = 0; % Regulation parameter have a value of zero
            
                    costFunction = @(p) nnCostFunction(p, ...
                                                       input_layer_size, ...
                                                       hidden_layer_size, ...
                                                       num_labels, xtrain, ytrain, lambda); % The cost function is calculated, also the gradient and the error matrix that allow to updating the Theta1 and Theta2 Values 
            
                    [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); % This function update the Theta1 and Theta2 values that generate the minimum cost (local minimum)
            
                    Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                                     hidden_layer_size, (input_layer_size + 1));
            
                    Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                                     num_labels, (hidden_layer_size + 1)); 
                    net.T1 = Theta1;
                    net.T2 = Theta2;
                    End_opt2(kfold,:) = toc(Start_opt2);

                    % Predict the testing values
                    pred_train = predict(Theta1, Theta2, xtrain);
                    Acc_tr = sum(ytrain==pred_train)/size(pred_train,1);
                    cc_tr = confusionmat(ytrain,pred_train);
                    Fpr_tr = cc_tr(2,1)/(cc_tr(2,1)+cc_tr(2,2));

                    % Predict the testing values
                    pred_test = predict(Theta1, Theta2, xtest);
                    Acc_te = sum(ytest==pred_test)/size(pred_test,1);
                    cc_te = confusionmat(ytest,pred_test);
                    Fpr_te = cc_te(2,1)/(cc_te(2,1)+cc_te(2,2));

                    % Guardamos las mejores redes
                    if Acc_te >= Acc_base 
                          r = Inic;
                          Acc_base = Acc_te;
                          red = net;
                    end
                    % Guardamos los resultados
                    Acc_kfold(kfold,:) = [Acc_tr Acc_te];
                    Fpr_kfold(kfold,:) = [Fpr_tr Fpr_te];
                end
                Acc_Increment(Inic,:) = mean(Acc_kfold);
                Fpr_Increment(Inic,:) = mean(Fpr_kfold);
                End_opt1(Inic,:) = toc(Start_opt1);
                End_opt2Final(Inic,:) = mean(End_opt2);
            end
            Time1Opt(:,Hidd) = End_opt1;
            Time2Opt(:,Hidd) = End_opt2Final;
            Result(Runs1).AccHidd{Hidd} = Acc_Increment;
            Result(Runs1).FprHidd{Hidd} = Fpr_Increment;
            Result(Runs1).NetHidd(Hidd).net = red;
        end
        
        clear Theta1 Theta2 initial_Theta1 initial_Theta2 options costFunction nn_params cost ...
              xtrain ytrain xtest ytrain Features Class Label

        % Se evalua cual es el mejor modelo
        for h = 1:10
            Val = Result(Runs1).AccHidd{h};
            Best_tr(h) = max(Val(:,1));
            Best_ts(h) = max(Val(:,2));
        end
        Best_better_tr = find(Best_tr==max(Best_tr));
        Best_better_ts = find(Best_ts==max(Best_ts));
        red_tr = Result(Runs1).NetHidd(Best_better_tr).net;
        red_ts = Result(Runs1).NetHidd(Best_better_ts).net;
        Theta_RT1 = red_ts.T1; 
        Theta_RT2 = red_ts.T2;
        [~,hidd] = max(Best_ts);
        Count = 1;

        for Runs2 = 1:5
            if Runs2~=Runs1
                
               % Define features and classes matrix to train 
               Features = [DataPSD(Runs2).cross; DataPSD(Runs2).right];
               Class = [ones(size(DataPSD(Runs2).cross,1),1)*1; ones(size(DataPSD(Runs2).right,1),1)*2];
               Label = Class;
               
               for kfold = 1:5 

                   % Data
                   xtrain = Features(~hpartition.test(kfold),:);
                   ytrain = Label(~hpartition.test(kfold),:);
                   xtest = Features(hpartition.test(kfold),:);
                   ytest = Label(hpartition.test(kfold),:);
    
                   % Adjust data
                   xtest = (xtest - mean(xtest))./std(xtest); % Normalized Z-score
                   xtrain = (xtrain - mean(xtrain))./std(xtrain); % Normalized Z-score

                   % Predict the testing values without retraining
                   pred_test_sin = predict(red_ts.T1, red_ts.T2, xtest);
                   Acc_sin_te(kfold) = sum(ytest==pred_test_sin)/size(pred_test_sin,1);
                   cc_sin_te = confusionmat(ytest,pred_test_sin);
                   Fpr_sin_te(kfold) = cc_sin_te(2,1)/(cc_sin_te(2,1)+cc_sin_te(2,2));
                    
                   Start_inc1 = tic;
                   for Ins = 1:100
                        
                        % Parameters
                        input_layer_size  = size(xtrain,2); % Input layer size correspond to the size of the used features 
                        hidden_layer_size = hidd; % Hidden layer can change, but in this ocassion have a value of 10 
                        num_labels = max(ytrain); % The output layer correspond to the predicted class, for this case are 6 classes
                        
                        % Inizializing parameters
                        Start_inc2 = tic;
                        initial_nn_params = [Theta_RT1(:); Theta_RT2(:)];
                        
                        options = optimset('MaxIter', 50); % The maximum number of iterations is established at 50
                
                        lambda = 0; % Regulation parameter have a value of zero
                
                        costFunction = @(p) nnCostFunction(p, ...
                                                           input_layer_size, ...
                                                           hidden_layer_size, ...
                                                           num_labels, xtrain, ytrain, lambda); % The cost function is calculated, also the gradient and the error matrix that allow to updating the Theta1 and Theta2 Values 
                
                        [nn_params, cost] = fmincg(costFunction, initial_nn_params, options); % This function update the Theta1 and Theta2 values that generate the minimum cost (local minimum)
                
                        Theta_RT1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                                         hidden_layer_size, (input_layer_size + 1));
                
                        Theta_RT2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                                         num_labels, (hidden_layer_size + 1));
                        End_inc2(Ins,:) = toc(Start_inc2);

                        % Predict the testing values
                        pred_test_con = predict(Theta_RT1, Theta_RT2, xtest);
                        Acc_con_te(Ins,:) = sum(ytest==pred_test_con)/size(pred_test_con,1);
                        cc_con_te = confusionmat(ytest,pred_test_con);
                        Fpr_con_te(Ins,:) = cc_con_te(2,1)/(cc_con_te(2,1)+cc_con_te(2,2));
                   end
                   Acc_rtraining(:,kfold) = Acc_con_te;
                   Fpr_rtraining(:,kfold) = Fpr_con_te;
                   End_inc1(:,kfold) = toc(Start_inc1);
                   End_inc2Final(:,kfold) = End_inc2;
%                    Theta_RT1 = red_ts.T1; 
%                    Theta_RT2 = red_ts.T2;
               end
               Time1 = mean(End_inc1);
               Time2 = mean(End_inc2Final,2);
               Retrining(Count).ConAcc_ins = mean(Acc_rtraining,2);
               Retrining(Count).ConFpr_ins = mean(Fpr_rtraining,2);
               Retrining(Count).SinAcc_ins = mean(Acc_sin_te,2);
               Retrining(Count).SinFpr_ins = mean(Fpr_sin_te,2);
               Count = Count + 1;
            end
        end
        TimeRun(:,Runs1) = Time2;
        ResultsHiddFin = Result;
        RetriningFin{Runs1} = Retrining;
    end
    display(['Sujeto ',num2str(Sujetos)])
    Metrics.Retrain = RetriningFin;
    Metrics.Normal = ResultsHiddFin;
%     save(['Result_Sub',num2str(Sujetos)],'Metrics')
end


