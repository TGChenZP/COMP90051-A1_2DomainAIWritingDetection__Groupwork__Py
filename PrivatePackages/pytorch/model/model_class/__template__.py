from environment import *

class ClassificationModel(object):
    """ Model Template for Classification """

    class GeneralModel():
        def __init__(self, configs):
            pass

    def __init__(self, configs, name="Model"):
        super().__init__()
        self.configs = configs
        self.name = self.configs.name 
        self.model = self.Model(self.configs) # create the model
        
        self.n_unique_tokens = self.configs.n_unique_tokens

        # operations
            # set seed - control randomness
        torch.manual_seed(self.configs.random_state) # set seed

            # optimiser and criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs.lr)
        self.criterion = self.configs.loss
        self.validation_criterion = self.configs.validation_loss
        self.pretrain_criterion = self.configs.pretrain_loss
        self.pretrain_validation_criterion = self.configs.pretrain_validation_loss

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
            # --- 
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device)

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X, total_train_y, total_train_domain, total_val_X, total_val_y, total_val_domain):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        # min_loss = np.inf
        max_bal_accuracy = 0
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.FloatTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X), y 

                # calculate loss
                loss = self.criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1
            
        
            epoch_loss /= n_batch

            # print epoch training results
            epoch_pred_label = [1 if i[1] > i[0] else 0 for i in epoch_pred]
            epoch_label = [1 if i[1] > i[0] else 0 for i in epoch_true]

            epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_f1 = f1_score(epoch_label, epoch_pred_label)
            epoch_bal_accu = balanced_accuracy_score(epoch_label, epoch_pred_label)

            epoch_pred_label_dom1 = []
            epoch_label_dom1 = []
            epoch_pred_label_dom2 = []
            epoch_label_dom2 = []
            for i in range(len(total_train_domain)):
                if total_train_domain[i][1] == 0:
                    epoch_label_dom1.append(epoch_label[i])
                    epoch_pred_label_dom1.append(epoch_pred_label[i])
                else:
                    epoch_label_dom2.append(epoch_label[i])
                    epoch_pred_label_dom2.append(epoch_pred_label[i])

            epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                    Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                    Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
            print(record)

            # Validation
            valid_loss, valid_bal_accu = self.eval(total_val_X, total_val_y, total_val_domain, epoch)
            self.model.train()
            # if valid_loss < min_loss:
            #     min_loss = valid_loss
            #     best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
            #     self.save()

            if valid_bal_accu > max_bal_accuracy:
                max_bal_accuracy = valid_bal_accu
                best_epoch = epoch
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def predict(self, future_X):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y
        

    def eval(self, total_val_X, total_val_y, total_val_domain, epoch, evaluation_mode = False):
        
        pred_val_y = self.predict(total_val_X)

        val_y = np.array(total_val_y)

        epoch_pred_y_label = [1 if i[1] > i[0] else 0 for i in pred_val_y]

        epoch_y_label = [1 if i[1] > i[0] else 0 for i in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        
        epoch_loss = self.validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)
        epoch_f1 = f1_score(epoch_y_label, epoch_pred_y_label)
        epoch_bal_accu = balanced_accuracy_score(epoch_y_label, epoch_pred_y_label)

        epoch_pred_label_dom1 = []
        epoch_label_dom1 = []
        epoch_pred_label_dom2 = []
        epoch_label_dom2 = []
        for i in range(len(total_val_domain)):
            if total_val_domain[i][1] == 0:
                epoch_label_dom1.append(epoch_y_label[i])
                epoch_pred_label_dom1.append(epoch_pred_y_label[i])
            else:
                epoch_label_dom2.append(epoch_y_label[i])
                epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss, epoch_bal_accu
    
    def fit_pretrain(self, total_train_X, total_train_y, total_train_domain, total_train_mask, total_val_X, total_val_y, total_val_domain, total_val_mask):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        min_loss = np.inf
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_mask)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y, mask = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                                torch.LongTensor(total_train_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X, mask), y

                # calculate loss
                loss = self.pretrain_criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1


        
            epoch_loss /= n_batch

            # # print epoch training results
            # epoch_pred_label = [np.argmax(np.array(pred)) for pred in epoch_pred]
            # epoch_label = [y for y in epoch_true]

            # epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)

            # epoch_pred_label_dom1 = []
            # epoch_label_dom1 = []
            # epoch_pred_label_dom2 = []
            # epoch_label_dom2 = []
            # for i in range(len(total_train_domain)):
            #     if total_train_domain[i] == 0:
            #         epoch_label_dom1.append(epoch_label[i])
            #         epoch_pred_label_dom1.append(epoch_pred_label[i])
            #     else:
            #         epoch_label_dom2.append(epoch_label[i])
            #         epoch_pred_label_dom2.append(epoch_pred_label[i])

            # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} '''
            # | Accuracy: {epoch_accuracy:>7.4f}| Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
            print(record)

            # Validation
            valid_loss = self.eval_pretrain(total_val_X, total_val_y, total_val_domain, total_val_mask, epoch)
            self.model.train()
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def eval_pretrain(self, total_val_X, total_val_y, total_val_domain, total_val_mask, epoch, evaluation_mode = False):
        pred_val_y = self.predict_pretrain(total_val_X, total_val_mask)

        val_y = np.array(total_val_y)

        # epoch_pred_y_label = [np.argmax(np.array(pred)) for pred in pred_val_y]
        # epoch_y_label = [y for y in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.LongTensor(val_y).to(self.device)

        epoch_loss = self.pretrain_validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        # epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)

        # epoch_pred_label_dom1 = []
        # epoch_label_dom1 = []
        # epoch_pred_label_dom2 = []
        # epoch_label_dom2 = []
        # for i in range(len(total_val_domain)):
        #     if total_val_domain[i] == 0:
        #         epoch_label_dom1.append(epoch_y_label[i])
        #         epoch_pred_label_dom1.append(epoch_pred_y_label[i])
        #     else:
        #         epoch_label_dom2.append(epoch_y_label[i])
        #         epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} '''
        # | Accuracy: {epoch_accuracy:>7.4f} | 
                            # Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss
    
    def predict_pretrain(self, future_X, future_mask):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X, mask = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(future_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X, mask)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y


class DANN_Model(object):
    """ Model Template for Classification """

    class GeneralModel():
        def __init__(self, configs):
            pass

    def __init__(self, configs, name="Model"):
        super().__init__()
        self.configs = configs
        self.name = self.configs.name 
        self.model = self.Model(self.configs) # create the model
        
        self.n_unique_tokens = self.configs.n_unique_tokens

        # operations
            # set seed - control randomness
        torch.manual_seed(self.configs.random_state) # set seed

            # optimiser and criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs.lr)
        self.criterion = self.configs.loss
        self.validation_criterion = self.configs.validation_loss
        self.domain_criterion = self.configs.domain_loss

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
            # --- 
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device)
        self.domain_criterion.to(self.device)

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X, total_train_y, total_train_domain, total_val_X, total_val_y, total_val_domain):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        # min_loss = np.inf
        max_bal_accuracy = 0
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_classifcation_loss = 0
            epoch_domain_loss = 0
            epoch_pred, epoch_true = [], []
            epoch_dom_pred, epoch_dom_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)

            n_batch = 0

            dont_use_gradient_reversal = epoch % self.configs.gradient_reversal_every_n_epoch

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y, dom = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.FloatTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.FloatTensor(total_train_domain[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                (pred, dom_pred), true = self.model(X), y 

                # calculate loss
                classification_loss = self.criterion(pred, true) 
                domain_loss = self.configs.alpha * self.domain_criterion(dom_pred, dom)
                if dont_use_gradient_reversal:
                    domain_loss *= 0
                loss = classification_loss + domain_loss

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_classifcation_loss += classification_loss.detach().cpu().numpy()
                epoch_domain_loss += domain_loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()
                epoch_dom_pred += dom_pred.detach().cpu().tolist()
                epoch_dom_true += dom.detach().cpu().tolist()

                n_batch += 1
            
            epoch_loss /= n_batch
            epoch_classifcation_loss /= n_batch
            epoch_domain_loss /= n_batch

            # print epoch training results
            epoch_pred_label = [1 if i[1] > i[0] else 0 for i in epoch_pred]
            epoch_label = [1 if i[1] > i[0] else 0 for i in epoch_true]

            epoch_dom_pred_label = [1 if i[1] > i[0] else 0 for i in epoch_dom_pred]
            epoch_dom_label = [1 if i[1] > i[0] else 0 for i in epoch_dom_true]

            epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_dom_accuracy = accuracy_score(epoch_dom_label, epoch_dom_pred_label)
            epoch_f1 = f1_score(epoch_label, epoch_pred_label)
            epoch_bal_accu = balanced_accuracy_score(epoch_label, epoch_pred_label)


            epoch_pred_label_dom1 = []
            epoch_label_dom1 = []
            epoch_pred_label_dom2 = []
            epoch_label_dom2 = []
            for i in range(len(total_train_domain)):
                if total_train_domain[i][1] == 0:
                    epoch_label_dom1.append(epoch_label[i])
                    epoch_pred_label_dom1.append(epoch_pred_label[i])
                else:
                    epoch_label_dom2.append(epoch_label[i])
                    epoch_pred_label_dom2.append(epoch_pred_label[i])

            epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Classification Loss: {epoch_classifcation_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                            Domain Loss: {epoch_domain_loss:>7.4f} | Domain Accuracy: {epoch_dom_accuracy:>7.4f} | 
                            Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                            Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
            print(record)

            # Validation
            valid_loss, valid_bal_accu = self.eval(total_val_X, total_val_y, total_val_domain, epoch)
            self.model.train()
            # if valid_loss < min_loss:
            #     min_loss = valid_loss
            #     best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
            #     self.save()

            if valid_bal_accu > max_bal_accuracy:
                max_bal_accuracy = valid_bal_accu
                best_epoch = epoch
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def predict(self, future_X):
        self.model.eval()

        pred_y = []
        pred_dom = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred, dom_pred = self.model(X)

                pred_y.extend(pred.detach().cpu().tolist())
                pred_dom.extend(dom_pred.detach().cpu().tolist())
    
        return pred_y, pred_dom
        

    def eval(self, total_val_X, total_val_y, total_val_domain, epoch, evaluation_mode = False):
        
        pred_val_y, pred_val_dom = self.predict(total_val_X)

        val_y = np.array(total_val_y)
        val_y_domain = np.array(total_val_domain)

        epoch_pred_y_label = [1 if i[1] > i[0] else 0 for i in pred_val_y]
        epoch_pred_dom_label = [1 if i[1] > i[0] else 0 for i in pred_val_dom]

        epoch_y_label = [1 if i[1] > i[0] else 0 for i in val_y]
        epoch_dom_label = [1 if i[1] > i[0] else 0 for i in val_y_domain]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        pred_val_dom_tensor = torch.FloatTensor(np.array(pred_val_dom)).to(self.device)
        val_dom_tensor = torch.FloatTensor(val_y_domain).to(self.device)
        
        epoch_loss = self.validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        epoch_dom_loss = self.domain_criterion(pred_val_dom_tensor, val_dom_tensor).cpu().numpy()
        epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)
        epoch_dom_accuracy = accuracy_score(epoch_dom_label, epoch_pred_dom_label)
        epoch_f1 = f1_score(epoch_y_label, epoch_pred_y_label)
        epoch_bal_accu = balanced_accuracy_score(epoch_y_label, epoch_pred_y_label)

        epoch_pred_label_dom1 = []
        epoch_label_dom1 = []
        epoch_pred_label_dom2 = []
        epoch_label_dom2 = []
        for i in range(len(total_val_domain)):
            if total_val_domain[i][1] == 0:
                epoch_label_dom1.append(epoch_y_label[i])
                epoch_pred_label_dom1.append(epoch_pred_y_label[i])
            else:
                epoch_label_dom2.append(epoch_y_label[i])
                epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Classification Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                            Domain Loss: {epoch_dom_loss:>7.4f} | Domain Accuracy: {epoch_dom_accuracy:>7.4f} |  
                            Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} |  
                            Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
        
        print(record)

        if not evaluation_mode:

            return epoch_loss, epoch_bal_accu
        

class DomainBCEModel(object):
    """ Model Template for Classification """

    class GeneralModel():
        def __init__(self, configs):
            pass

    def __init__(self, configs, name="Model"):
        super().__init__()
        self.configs = configs
        self.name = self.configs.name 
        self.model = self.Model(self.configs) # create the model
        
        self.n_unique_tokens = self.configs.n_unique_tokens

        # operations
            # set seed - control randomness
        torch.manual_seed(self.configs.random_state) # set seed

            # optimiser and criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs.lr)
        self.dom_1_criterion = self.configs.domain_1_loss
        self.dom_1_validation_criterion = self.configs.domain_1_validation_loss
        self.dom_2_criterion = self.configs.domain_2_loss
        self.dom_2_validation_criterion = self.configs.domain_2_validation_loss
        self.domain_prior = self.configs.domain_prior
        self.pretrain_criterion = self.configs.pretrain_loss
        self.pretrain_validation_criterion = self.configs.pretrain_validation_loss
        self.criterion = self.configs.loss
        self.validation_criterion = self.configs.validation_loss

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
            # --- 
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device)
        self.dom_1_criterion.to(self.device)
        self.dom_2_criterion.to(self.device)
        self.dom_1_validation_criterion.to(self.device)
        self.dom_2_validation_criterion.to(self.device)

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X, total_train_y, total_train_domain, total_val_X, total_val_y, total_val_domain):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        # min_loss = np.inf
        max_bal_accuracy = 0
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
                
                neg_dom = torch.BoolTensor([1 if x[1] == 0 else 0 for x in total_train_domain[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]])
                pos_dom = torch.BoolTensor([1 if x[1] == 1 else 0 for x in total_train_domain[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]])
                
                X, y = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.FloatTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X), y

                neg_dom_percentage = neg_dom.sum()/self.configs.batch_size
                pos_dom_percentage = pos_dom.sum()/self.configs.batch_size

                # calculate loss
                domain_1_loss = self.dom_1_criterion(pred[neg_dom], true[neg_dom])
                domain_2_loss = self.dom_2_criterion(pred[pos_dom], true[pos_dom]) 

                loss = self.domain_prior[0] * (domain_1_loss if neg_dom_percentage else 0) * neg_dom_percentage \
                            + self.domain_prior[1] * (domain_2_loss if pos_dom_percentage else 0) * pos_dom_percentage

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()
                n_batch += 1

            epoch_loss /= n_batch

            # print epoch training results
            epoch_pred_label = [1 if i[1] > i[0] else 0 for i in epoch_pred]
            epoch_label = [1 if i[1] > i[0] else 0 for i in epoch_true]

            epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_f1 = f1_score(epoch_label, epoch_pred_label)
            epoch_bal_accu = balanced_accuracy_score(epoch_label, epoch_pred_label)

            epoch_pred_label_dom1 = []
            epoch_label_dom1 = []
            epoch_pred_label_dom2 = []
            epoch_label_dom2 = []
            for i in range(len(total_train_domain)):
                if total_train_domain[i][1] == 0:
                    epoch_label_dom1.append(epoch_label[i])
                    epoch_pred_label_dom1.append(epoch_pred_label[i])
                else:
                    epoch_label_dom2.append(epoch_label[i])
                    epoch_pred_label_dom2.append(epoch_pred_label[i])

            epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                    Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                    Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
            print(record)

            # Validation
            valid_loss, valid_bal_accu = self.eval(total_val_X, total_val_y, total_val_domain, epoch)
            self.model.train()
            # if valid_loss < min_loss:
            #     min_loss = valid_loss
            #     best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
            #     self.save()

            if valid_bal_accu > max_bal_accuracy:
                max_bal_accuracy = valid_bal_accu
                best_epoch = epoch
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def predict(self, future_X):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y
        

    def eval(self, total_val_X, total_val_y, total_val_domain, epoch, evaluation_mode = False):
        
        pred_val_y = self.predict(total_val_X)

        val_y = np.array(total_val_y)

        epoch_pred_y_label = [1 if i[1] > i[0] else 0 for i in pred_val_y]

        epoch_y_label = [1 if i[1] > i[0] else 0 for i in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        
        epoch_loss = self.validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)
        epoch_f1 = f1_score(epoch_y_label, epoch_pred_y_label)
        epoch_bal_accu = balanced_accuracy_score(epoch_y_label, epoch_pred_y_label)

        epoch_pred_label_dom1 = []
        epoch_label_dom1 = []
        epoch_pred_label_dom2 = []
        epoch_label_dom2 = []
        for i in range(len(total_val_domain)):
            if total_val_domain[i][1] == 0:
                epoch_label_dom1.append(epoch_y_label[i])
                epoch_pred_label_dom1.append(epoch_pred_y_label[i])
            else:
                epoch_label_dom2.append(epoch_y_label[i])
                epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss, epoch_bal_accu
    
    def fit_pretrain(self, total_train_X, total_train_y, total_train_domain, total_train_mask, total_val_X, total_val_y, total_val_domain, total_val_mask):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        min_loss = np.inf
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_mask)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y, mask = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                                torch.LongTensor(total_train_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X, mask), y

                # calculate loss
                loss = self.pretrain_criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1


        
            epoch_loss /= n_batch

            # # print epoch training results
            # epoch_pred_label = [np.argmax(np.array(pred)) for pred in epoch_pred]
            # epoch_label = [y for y in epoch_true]

            # epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)

            # epoch_pred_label_dom1 = []
            # epoch_label_dom1 = []
            # epoch_pred_label_dom2 = []
            # epoch_label_dom2 = []
            # for i in range(len(total_train_domain)):
            #     if total_train_domain[i] == 0:
            #         epoch_label_dom1.append(epoch_label[i])
            #         epoch_pred_label_dom1.append(epoch_pred_label[i])
            #     else:
            #         epoch_label_dom2.append(epoch_label[i])
            #         epoch_pred_label_dom2.append(epoch_pred_label[i])

            # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} '''
            # | Accuracy: {epoch_accuracy:>7.4f}| Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
            print(record)

            # Validation
            valid_loss = self.eval_pretrain(total_val_X, total_val_y, total_val_domain, total_val_mask, epoch)
            self.model.train()
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def eval_pretrain(self, total_val_X, total_val_y, total_val_domain, total_val_mask, epoch, evaluation_mode = False):
        pred_val_y = self.predict_pretrain(total_val_X, total_val_mask)

        val_y = np.array(total_val_y)

        # epoch_pred_y_label = [np.argmax(np.array(pred)) for pred in pred_val_y]
        # epoch_y_label = [y for y in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.LongTensor(val_y).to(self.device)

        epoch_loss = self.pretrain_validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        # epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)

        # epoch_pred_label_dom1 = []
        # epoch_label_dom1 = []
        # epoch_pred_label_dom2 = []
        # epoch_label_dom2 = []
        # for i in range(len(total_val_domain)):
        #     if total_val_domain[i] == 0:
        #         epoch_label_dom1.append(epoch_y_label[i])
        #         epoch_pred_label_dom1.append(epoch_pred_y_label[i])
        #     else:
        #         epoch_label_dom2.append(epoch_y_label[i])
        #         epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} '''
        # | Accuracy: {epoch_accuracy:>7.4f} | 
                            # Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss
    
    def predict_pretrain(self, future_X, future_mask):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X, mask = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(future_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X, mask)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y
    


class HingeModel(object):
    """ Model Template for Classification """

    class GeneralModel():
        def __init__(self, configs):
            pass

    def __init__(self, configs, name="Model"):
        super().__init__()
        self.configs = configs
        self.name = self.configs.name 
        self.model = self.Model(self.configs) # create the model
        
        self.n_unique_tokens = self.configs.n_unique_tokens

        # operations
            # set seed - control randomness
        torch.manual_seed(self.configs.random_state) # set seed

            # optimiser and criterion
        self.optimizer = AdamW(self.model.parameters(), lr=self.configs.lr)
        self.criterion = self.configs.loss
        self.validation_criterion = self.configs.validation_loss
        self.pretrain_criterion = self.configs.pretrain_loss
        self.pretrain_validation_criterion = self.configs.pretrain_validation_loss

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
            # --- 
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device)

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X, total_train_y, total_train_domain, total_val_X, total_val_y, total_val_domain):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        # min_loss = np.inf
        max_bal_accuracy = 0
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.FloatTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X), y 

                # calculate loss
                loss = self.criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1
            
        
            epoch_loss /= n_batch

            # print epoch training results
            epoch_pred_label = [1 if i[0] > 0 else 0 for i in epoch_pred]
            epoch_label = [1 if i[0] > 0 else 0 for i in epoch_true]

            epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_f1 = f1_score(epoch_label, epoch_pred_label)
            epoch_bal_accu = balanced_accuracy_score(epoch_label, epoch_pred_label)

            epoch_pred_label_dom1 = []
            epoch_label_dom1 = []
            epoch_pred_label_dom2 = []
            epoch_label_dom2 = []
            for i in range(len(total_train_domain)):
                if total_train_domain[i][1] == 0:
                    epoch_label_dom1.append(epoch_label[i])
                    epoch_pred_label_dom1.append(epoch_pred_label[i])
                else:
                    epoch_label_dom2.append(epoch_label[i])
                    epoch_pred_label_dom2.append(epoch_pred_label[i])

            epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
            epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                    Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                    Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
            print(record)

            # Validation
            valid_loss, valid_bal_accu = self.eval(total_val_X, total_val_y, total_val_domain, epoch)
            self.model.train()
            # if valid_loss < min_loss:
            #     min_loss = valid_loss
            #     best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
            #     self.save()

            if valid_bal_accu > max_bal_accuracy:
                max_bal_accuracy = valid_bal_accu
                best_epoch = epoch
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def predict(self, future_X):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y
        

    def eval(self, total_val_X, total_val_y, total_val_domain, epoch, evaluation_mode = False):
        
        pred_val_y = self.predict(total_val_X)

        val_y = np.array(total_val_y)

        epoch_pred_y_label = [1 if i[0] > 0 else 0 for i in pred_val_y]

        epoch_y_label = [1 if i[0] > 0 else 0 for i in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        
        epoch_loss = self.validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)
        epoch_f1 = f1_score(epoch_y_label, epoch_pred_y_label)
        epoch_bal_accu = balanced_accuracy_score(epoch_y_label, epoch_pred_y_label)

        epoch_pred_label_dom1 = []
        epoch_label_dom1 = []
        epoch_pred_label_dom2 = []
        epoch_label_dom2 = []
        for i in range(len(total_val_domain)):
            if total_val_domain[i][1] == 0:
                epoch_label_dom1.append(epoch_y_label[i])
                epoch_pred_label_dom1.append(epoch_pred_y_label[i])
            else:
                epoch_label_dom2.append(epoch_y_label[i])
                epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_f1_dom1 = f1_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_bal_accu_dom1 = balanced_accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_f1_dom2 = f1_score(epoch_label_dom2, epoch_pred_label_dom2)
        epoch_bal_accu_dom2 = balanced_accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Dom Avg Accuracy: {(epoch_bal_accu_dom1+epoch_bal_accu_dom2)/2:>7.4f} |
                Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f}| Domain 1 F1: {epoch_f1_dom1:>7.4f} | Domain 1 Balanced Accuracy: {epoch_bal_accu_dom1:>7.4f} | 
                Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| Domain 2 F1: {epoch_f1_dom2:>7.4f} | Domain 2 Balanced Accuracy: {epoch_bal_accu_dom2:>7.4f}'''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss, epoch_bal_accu
    
    def fit_pretrain(self, total_train_X, total_train_y, total_train_domain, total_train_mask, total_val_X, total_val_y, total_val_domain, total_val_mask):
        
        total_train_X, total_train_y = copy.deepcopy(total_train_X), copy.deepcopy(total_train_y)

        self.model.train()

        # scheduler and patience
        scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=0.5, patience=self.configs.patience//2) if self.configs.scheduler else None
        patience = self.configs.patience

        # set random seed
        np.random.seed(self.configs.random_state) # TODO: if use dataset and dataloader, need to change
        seeds = [np.random.randint(0, 1000000) for _ in range(self.configs.epochs)]

        min_loss = np.inf
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_mask)


            n_batch = 0

            for mini_batch_number in tqdm(range(len(total_train_X)//self.configs.batch_size+1)):
 
                
                X, y, mask = torch.LongTensor(total_train_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(total_train_y[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                                torch.LongTensor(total_train_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)
                
                if len(X) == 0:
                    break

                self.optimizer.zero_grad()
                pred, true = self.model(X, mask), y

                # calculate loss
                loss = self.pretrain_criterion(pred, true)

                # backpropagation
                loss.backward()
                if self.configs.grad_clip: # gradient clip
                    nn.utils.clip_grad_norm(self.model.parameters(), 2)
                self.optimizer.step()
                
                epoch_loss += loss.detach().cpu().numpy()
                epoch_pred += pred.detach().cpu().tolist()
                epoch_true += true.detach().cpu().tolist()

                n_batch += 1


        
            epoch_loss /= n_batch

            # # print epoch training results
            # epoch_pred_label = [np.argmax(np.array(pred)) for pred in epoch_pred]
            # epoch_label = [y for y in epoch_true]

            # epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)

            # epoch_pred_label_dom1 = []
            # epoch_label_dom1 = []
            # epoch_pred_label_dom2 = []
            # epoch_label_dom2 = []
            # for i in range(len(total_train_domain)):
            #     if total_train_domain[i] == 0:
            #         epoch_label_dom1.append(epoch_label[i])
            #         epoch_pred_label_dom1.append(epoch_pred_label[i])
            #     else:
            #         epoch_label_dom2.append(epoch_label[i])
            #         epoch_pred_label_dom2.append(epoch_pred_label[i])

            # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
            # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

            record = f'''Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} '''
            # | Accuracy: {epoch_accuracy:>7.4f}| Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
            print(record)

            # Validation
            valid_loss = self.eval_pretrain(total_val_X, total_val_y, total_val_domain, total_val_mask, epoch)
            self.model.train()
            if valid_loss < min_loss:
                min_loss = valid_loss
                best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
                self.save()

            else:
                patience -= 1
            if scheduler:
                scheduler.step(valid_loss)

        return best_epoch

    def eval_pretrain(self, total_val_X, total_val_y, total_val_domain, total_val_mask, epoch, evaluation_mode = False):
        pred_val_y = self.predict_pretrain(total_val_X, total_val_mask)

        val_y = np.array(total_val_y)

        # epoch_pred_y_label = [np.argmax(np.array(pred)) for pred in pred_val_y]
        # epoch_y_label = [y for y in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.LongTensor(val_y).to(self.device)

        epoch_loss = self.pretrain_validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        # epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)

        # epoch_pred_label_dom1 = []
        # epoch_label_dom1 = []
        # epoch_pred_label_dom2 = []
        # epoch_label_dom2 = []
        # for i in range(len(total_val_domain)):
        #     if total_val_domain[i] == 0:
        #         epoch_label_dom1.append(epoch_y_label[i])
        #         epoch_pred_label_dom1.append(epoch_pred_y_label[i])
        #     else:
        #         epoch_label_dom2.append(epoch_y_label[i])
        #         epoch_pred_label_dom2.append(epoch_pred_y_label[i])
                
        # epoch_accuracy_dom1 = accuracy_score(epoch_label_dom1, epoch_pred_label_dom1)
        # epoch_accuracy_dom2 = accuracy_score(epoch_label_dom2, epoch_pred_label_dom2)

        record = f'''Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} '''
        # | Accuracy: {epoch_accuracy:>7.4f} | 
                            # Domain 1 Accuracy: {epoch_accuracy_dom1:>7.4f} | Domain 2 Accuracy: {epoch_accuracy_dom2:>7.4f}| '''
       
        print(record)

        if not evaluation_mode:

            return epoch_loss
    
    def predict_pretrain(self, future_X, future_mask):
        self.model.eval()

        pred_y = []

        with torch.no_grad():

            for mini_batch_number in range(len(future_X)//self.configs.batch_size+1): 

                X, mask = torch.LongTensor(future_X[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device), \
                            torch.LongTensor(future_mask[mini_batch_number*self.configs.batch_size:(mini_batch_number+1)*self.configs.batch_size]).to(self.device)

                if len(X) == 0:
                    break
                
                pred = self.model(X, mask)

                pred_y.extend(pred.detach().cpu().tolist())
    
        return pred_y