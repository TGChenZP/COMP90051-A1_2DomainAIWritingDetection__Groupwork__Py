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

        # automatically detect GPU device if avilable
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.configs.device = self.device
            # --- 
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.validation_criterion.to(self.device)

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

        self.training_record = []

    def __str__(self):
        return self.name
    
    def save(self, mark=''):
        mark = ' ' + mark if mark else mark
        torch.save(self.model.state_dict(), os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'))
    
    def load(self, mark=''):
        mark = ' ' + mark if mark else mark
        self.model.load_state_dict(torch.load(os.path.join(self.configs.rootpath, f'state/{self}{mark}.pt'), map_location=self.device))
    
    def fit(self, total_train_X, total_train_y, total_val_X, total_val_y):
        
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
            record = f'Epoch {epoch+1} Train | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} '
            print(record)
            self.training_record.append(record)

            # Validation
            valid_loss = self.eval(total_val_X, total_val_y, epoch)
            self.model.train()
            if valid_loss <= min_loss:
                min_loss = valid_loss
                best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
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
        

    def eval(self, total_val_X, total_val_y, epoch, evaluation_mode = False):
        
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
        record = f'Epoch {epoch+1} Val   | Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} '
        print(record)
        self.training_record.append(record) 

        if not evaluation_mode:

            return epoch_loss
        

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

        # self.regularisation_loss = self.configs.regularisation_loss if self.configs.regularisation_loss else None

        self.training_record = []

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

        min_loss = np.inf
        best_epoch = 0
        for epoch in range(self.configs.epochs):

            if not patience: # end training when no more patience
                break

            epoch_loss = 0
            epoch_classifcation_loss = 0
            epoch_domain_loss = 0
            epoch_pred, epoch_true = [], []
            

            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_X)
            np.random.seed(seeds[epoch]) # reset seed so that they are shuffled in same order
            np.random.shuffle(total_train_y)
            np.random.seed(seeds[epoch])
            np.random.shuffle(total_train_domain)

            n_batch = 0

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
                domain_loss = self.configs.alpha * self.domain_criterion(dom_pred, dom).to(self.device)
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

                n_batch += 1
            
        
            epoch_loss /= n_batch
            epoch_classifcation_loss /= n_batch
            epoch_domain_loss /= n_batch

            # print epoch training results
            epoch_pred_label = [1 if i[1] > i[0] else 0 for i in epoch_pred]
            epoch_label = [1 if i[1] > i[0] else 0 for i in epoch_true]

            epoch_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_dom_accuracy = accuracy_score(epoch_label, epoch_pred_label)
            epoch_f1 = f1_score(epoch_label, epoch_pred_label)
            epoch_bal_accu = balanced_accuracy_score(epoch_label, epoch_pred_label)
            record = f'Epoch {epoch+1} Train | Classification Loss: {epoch_classifcation_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Domain Loss: {epoch_domain_loss:>7.4f} | Domain Accuracy: {epoch_dom_accuracy:>7.4f}'
            print(record)
            self.training_record.append(record)

            # Validation
            valid_loss = self.eval(total_val_X, total_val_y, total_val_domain, epoch)
            self.model.train()
            if valid_loss <= min_loss:
                min_loss = valid_loss
                best_epoch = epoch # TODO: Note that if use this to retrain, need to +1 as it is 0-indexed
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
        val_y = np.array(total_val_domain)

        epoch_pred_y_label = [1 if i[1] > i[0] else 0 for i in pred_val_y]
        epoch_pred_dom_label = [1 if i[1] > i[0] else 0 for i in pred_val_dom]

        epoch_y_label = [1 if i[1] > i[0] else 0 for i in val_y]
        epoch_dom_label = [1 if i[1] > i[0] else 0 for i in val_y]

        pred_val_y_tensor = torch.FloatTensor(np.array(pred_val_y)).to(self.device)
        val_y_tensor = torch.FloatTensor(val_y).to(self.device)
        pred_val_dom_tensor = torch.FloatTensor(np.array(pred_val_dom)).to(self.device)
        val_dom_tensor = torch.FloatTensor(val_y).to(self.device)
        
        epoch_loss = self.validation_criterion(pred_val_y_tensor, val_y_tensor).cpu().numpy()
        epoch_dom_loss = self.domain_criterion(pred_val_dom_tensor, val_dom_tensor).cpu().numpy()
        epoch_accuracy = accuracy_score(epoch_y_label, epoch_pred_y_label)
        epoch_dom_accuracy = accuracy_score(epoch_dom_label, epoch_pred_dom_label)
        epoch_f1 = f1_score(epoch_y_label, epoch_pred_y_label)
        epoch_bal_accu = balanced_accuracy_score(epoch_y_label, epoch_pred_y_label)
        record = f'Epoch {epoch+1} Val   | Classification Loss: {epoch_loss:>7.4f} | Accuracy: {epoch_accuracy:>7.4f}| F1: {epoch_f1:>7.4f} | Balanced Accuracy: {epoch_bal_accu:>7.4f} | Domain Loss: {epoch_dom_loss:>7.4f} | Domain Accuracy: {epoch_dom_accuracy:>7.4f}'
        print(record)
        self.training_record.append(record) 

        if not evaluation_mode:

            return epoch_loss