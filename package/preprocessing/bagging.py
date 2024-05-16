from sklearn.decomposition import PCA

MAX_ITEM = 32
BAGGING_THRESHOLD = 0.9


class sparser_node:
    def __init__(self, words, k):
        self.k = k # the depth of tree, also the digit of current sparser
        self.name = words[0][:self.k] + "*" # name of this node
        self.bagged = False # if the node is a bag, determined after call bagging function
        
        items = []
        collector = {}
        # separate the words which is not long enough
        # for example, got words = ["a", "b", "a1", "a2"]
        # then the items of this bag should be ["a", "b"]
        # the collector of child will be ["a1", "a2"]
        for word in words:
            if len(word) > self.k+1:
                if word[self.k] not in collector:
                    collector[word[self.k]] = []
                collector[word[self.k]].append(word)
            else:
                items.append(word)
        
        self.items = items
        self.children = {}
        # ["a1", "a2"] => {"a": [a1, a2]}
        for symbol in collector:
            self.children[symbol] = sparser_node(collector[symbol], self.k+ 1)
        
    def __call__(self, idx):
        return self.children[idx]
    
    def bagging(self, df):
        items = []
        items += self.items
        
        for child in self.children:
            items+= self.children[child].bagging(df) # recurrently call bagging (deep first, digit tree)
        
        n = len(items) # only 1 item => no bagging
        if n == 1:
            return items
        elif n< MAX_ITEM: # max number of items for each bag
            data = df[items]
            ### bagging rules
            # rule 1
            #score = np.mean(data.corr().abs())
            # rule 2
            pca = PCA(n)
            tmp_data = pca.fit_transform((data-data.mean()/data.std()))
            score = pca.explained_variance_ratio_[0]
            
            if score> BAGGING_THRESHOLD:
                #print(np.mean(data.corr().abs()), pca.explained_variance_ratio_, "#################")
                #df[self.name] = df[items].mean(axis = 1) # act 1
                df[self.name] = tmp_data[:, 0] # act 1
                
                self.bagged = True # record this node is bagged
                return [self.name] # bag name
        return items 
        
    def collect(self, return_bag = True):
        if self.bagged and return_bag:
            return [self.name]

        items = []
        items += self.items
                
        for child in self.children:
            items+= self.children[child].collect(return_bag)
        return items
    
    def see(self, idx):
        if len(idx)>1:
            return self(idx[0]).see(idx[1:])
        if idx =="*":
            return self.collect(return_bag = False)
        else:
            for i in self.items:
                if i[-1] == idx:
                    return [i]

class bagger():
    def __init__(self):
        self.root = None
        self.fitted = False


        
    def fit(self, x, y= None):
        # sparse the column names
        self.root = sparser_node(x.columns, 0)

        # bagging
        self.template_x = x.copy()
        self.bags = self.root.bagging(self.template_x)
        
        self.fitted = True
        return self

    def transform(self, x, y= None):
        if not self.fitted:
            raise "please call fit before calling transform."

        return self.template_x[self.bags], y
        
    def fit_transform(self, x, y= None):
        self.fit(x, y)
        return self.transform(x, y)

    def see(self, idx):
        return self.root.see(idx)

    def unbagging(self, name):
        return " ".join(self.see(name))
