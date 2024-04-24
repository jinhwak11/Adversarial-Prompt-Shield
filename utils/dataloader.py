import pandas as pd
import json
class DataPreprocessing():
    '''
    ######
    Usage example
    training_data = "bbf_s,bbf_a,red"
    dataprep = DataPreprocessing(training_data, data_folder_path='/mnt/llm/data/')
    train_data = dataprep.read_and_processing('train')


     ######
    : Arguments
            data_folder_path (str): path where data is stored in, parent directory
            prep_datas  (str, sep=','): dataname to train the model (e.g. wtc,bbf_s,bbf_a,bbf_multi,bad)
            cols (str list) : DataFrame column names used to preprocess data; it should indicate a column contatining dialogue data, a column indicating label, and a column to indicate data name


    ######
    function read_and_processing():
        output : DataFrame with 3 columns (Default : transcript, rating, datatype)
            (str) transcript : dialogues between Human and Assistant. Each utterance is annotated with either Human: or Assistant:. deliminator : '\n\n'.
            (binary) rating : safe, unsafe
            datatype : data name (e.g. bbf)
    '''


    def __init__(self,dataname,data_folder_path='./data/',cols=['transcript','rating','datatype']):
        self.data_folder_path = data_folder_path
        self.prep_datas = dataname.split(',')
        self.cols = cols

    def read_and_processing(self,type):
        self.data = []
        print(f"TYPE: {type}")
        for dataname in self.prep_datas:
            print(f"PROCESSING {dataname}")
            if '_rand' in dataname:
                random = '_RS_20'
                dataname = dataname.split('_rand')[0]
                print("Data Name: ", dataname, "RANDOM: ", random)
            else:
                random = ''

            if dataname == 'red':
                print(f" data loading from {self.data_folder_path}/red-team/red-team-{type}{random}.csv")
                red_data = pd.read_csv(f"{self.data_folder_path}/red-team/red-team-{type}{random}.csv", index_col=[0])
                red_data["datatype"] = f"red{random}"
                self.data.append(red_data[self.cols])

            elif dataname == 'bbf_s':
                print(f" data loading from {self.data_folder_path}/dialogue_safety/single_turn_safety{random}.json")
                with open(f"{self.data_folder_path}/dialogue_safety/single_turn_safety{random}.json") as f:
                    bbf_single_data = json.load(f)
                bbf_df = pd.DataFrame()
                for i in range(1, 4):
                    bbf_standard_single_df = pd.concat([pd.DataFrame(bbf_single_data['standard'][type]['3']['bad']),
                                                        pd.DataFrame(bbf_single_data['standard'][type]['3']['good'])],
                                                       ignore_index=True)
                    bbf_df = pd.concat([bbf_df, bbf_standard_single_df], ignore_index=True)
                bbf_df['transcript'] = bbf_df['text'].map(lambda x: '\n\nHuman:' + x)
                bbf_df['rating'] = bbf_df['labels'].map(lambda x: 1 if 'not' in str(x[0]) else 0)
                bbf_df['datatype'] = f"bbf_s{random}"
                self.data.append(bbf_df[self.cols])

            elif dataname =='bbf_a': # bbf adversarial
                print(f" data loading from {self.data_folder_path}/dialogue_safety/single_turn_safety{random}.json")
                with open(f"{self.data_folder_path}/dialogue_safety/single_turn_safety{random}.json") as f:
                    bbf_single_data= json.load(f)
                bbf_df= pd.DataFrame()
                for i in range(1,4):
                    bbf_adversarial_single_df = pd.concat([pd.DataFrame(bbf_single_data['adversarial'][type][str(i)]['bad']),pd.DataFrame(bbf_single_data['adversarial'][type][str(i)]['good'])],ignore_index=True)
                    bbf_df = pd.concat([bbf_df,bbf_adversarial_single_df] , ignore_index=True)
                bbf_df['transcript'] = bbf_df['text'].map(lambda x: '\n\nHuman:' + x )
                bbf_df['rating'] = bbf_df['labels'].map(lambda x : 1 if 'not' in str(x[0]) else 0)
                bbf_df["datatype"] = f"bbf_a{random}"
                self.data.append(bbf_df[self.cols])

            elif dataname == 'bbf_multi':
                print(f" data loading from {self.data_folder_path}/dialogue_safety/multi_turn_safety{random}.json")
                with open(f"{self.data_folder_path}/dialogue_safety/multi_turn_safety{random}.json") as f:
                    bbf_data= json.load(f)
                bbf_data  = pd.DataFrame(bbf_data[type])
                bbf_data['transcript'] = bbf_data['text'].map(lambda x : '\n\n' + '\n\n'.join([ ("Human: " + text) if i%2==0 else ("Assistant: " + text) for i, text in enumerate(x.split('\n'))]))
                bbf_data['rating'] = bbf_data['labels'].map(lambda x : 1 if x[0]=='__notok__' else 0)
                bbf_data["datatype"] = f"bbf_multi{random}"
                self.data.append(bbf_data[self.cols])

            elif dataname =='bad': #bot-adversarial-dialogue dataset
                print(f" data loading from {self.data_folder_path}/bot_adversarial_dialogue_datasets_with_persona/{type}{random}.txt")
                with open(f"{self.data_folder_path}/bot_adversarial_dialogue_datasets_with_persona/{type}{random}.txt" ,"r") as f:
                    bad_data =  f.readlines()
                bad_data = [{i.split(':')[0]:i.split(':')[1] for i in d.split('\t')} for d in bad_data if len(d)>1]
                bad_data_df = pd.DataFrame(bad_data)
                bad_data_df['transcript'] = bad_data_df['text'].map(lambda x : '\n\n'+ '\n\n'.join([ ("Human: " + text) if i%2==0 else ("Assistant: " + text) for i, text in enumerate(x.split('\\n'))]))
                bad_data_df['rating'] = bad_data_df['labels'].map(lambda x :1 if '__notok__' in x  else 0)
                bad_data_df["datatype"] = f"bad{random}"
                self.data.append(bad_data_df[self.cols])

            elif dataname == 'wtc': #wikipedia-toxic-comment
                cols = ['toxic','severe_toxic',	'obscene',	'threat',	'insult','identity_hate']
                if type =='train' or type=='valid':
                    print(f" data loading from {self.data_folder_path}/wikipedia_toxic_comment/wtc_train{random}.csv")
                    wtc_data = pd.read_csv(f"{self.data_folder_path}/wikipedia_toxic_comment/wtc_train{random}.csv")
                    from sklearn.model_selection import train_test_split
                    wtc_train_data, wtc_valid_data = train_test_split(wtc_data,test_size=0.25,random_state=0)
                    if type =='train':
                        wtc_data = wtc_train_data.reset_index(drop=True)
                    elif type =='valid':
                        wtc_data = wtc_valid_data.reset_index(drop=True)
                    wtc_data = wtc_data[(wtc_data[cols]>-1).any(axis=1)].reset_index()
                    wtc_data['label'] = (wtc_data[cols]>0).any(axis=1)
                    wtc_data['transcript'] = wtc_data['comment_text'].map(lambda x : '\n\nHuman:' + x.replace('\n\n',' ') )
                    wtc_data['rating'] = wtc_data['label'].map(lambda x : int(x))
                    wtc_data["datatype"] = f"wtc{random}"
                    self.data.append(wtc_data[self.cols])
                elif type =='test':
                    print(f" data loading from {self.data_folder_path}/wikipedia_toxic_comment/wtc_test{random}.csv")
                    wtc_test_data = pd.read_csv(f"{self.data_folder_path}/wikipedia_toxic_comment/wtc_test{random}.csv")
                    wtc_test_label = pd.read_csv(f"{self.data_folder_path}/wikipedia_toxic_comment/wtc_test_labels.csv")
                    wtc_data = wtc_test_data.merge(wtc_test_label)
                    wtc_data = wtc_data[(wtc_data[cols]>-1).any(axis=1)].reset_index()
                    wtc_data['label'] = (wtc_data[cols]>0).any(axis=1)
                    wtc_data['transcript'] = wtc_data['comment_text'].map(lambda x : '\n\nHuman:' + x.replace('\n\n',' ') )
                    wtc_data['rating'] = wtc_data['label'].map(lambda x : int(x))
                    wtc_data["datatype"] = f"wtc{random}"
                    self.data.append(wtc_data[self.cols])

            elif dataname == 'llm-behavior':
                if random:
                    random = '_random'
                harmful_data = pd.read_csv(f"{self.data_folder_path}/llm-attack/harmful_behaviors{random}.csv")
                harmful_data['transcript'] = harmful_data['goal'].map(lambda x : '\n\nHuman:' + x )
                harmful_data['rating'] = 1
                harmful_data['datatype'] = f'llm-behavior{random}'
                self.data.append(harmful_data[self.cols])

            elif dataname == 'llm-attack':
                import pickle
                if type == 'train' or type =='valid':
                    print(f" data loading from {self.data_folder_path}/{dataname}/train_All_rnd.pkl")
                    with open(f'{self.data_folder_path}/{dataname}/train_All_rnd.pkl', 'rb') as file:
                        attack_loaded_data = pickle.load(file)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' +d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'llm-attack'
                    from sklearn.model_selection import train_test_split
                    harmful_train, harmful_valid = train_test_split(harmful_data, test_size=0.25, random_state=0)
                    if type =='train':
                        self.data.append(harmful_train[self.cols])
                    elif type =='valid':
                        self.data.append(harmful_valid[self.cols])

                elif type=='test':
                    print(f" data loading from {self.data_folder_path}/{dataname}/test_All_rnd.pkl")
                    with open(f'{self.data_folder_path}/{dataname}/test_All_rnd.pkl', 'rb') as file:
                        attack_loaded_data = pickle.load(file)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' +d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'llm-attack'
                    self.data.append(harmful_data[self.cols])
                    print(harmful_data.head())

            elif dataname == 'random-attack':
                if type == 'train' or type =='valid':
                    import pickle
                    print(f" data loading from {self.data_folder_path}/{dataname}/train_All_rnd.pkl")
                    with open(f'{self.data_folder_path}/{dataname}/train_All_rnd.pkl','rb') as f:
                        attack_loaded_data = pickle.load(f)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' +d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'random-attack'
                    from sklearn.model_selection import train_test_split
                    harmful_train, harmful_valid = train_test_split(harmful_data, test_size=0.25, random_state=0)
                    if type =='train':
                        self.data.append(harmful_train[self.cols])
                    elif type =='valid':
                        self.data.append(harmful_valid[self.cols])

                elif type =='test':
                    import pickle
                    print(f" data loading from {self.data_folder_path}/{dataname}/test_All_rnd.pkl")
                    with open(f'{self.data_folder_path}/llm-attack/test_All_rnd.pkl', 'rb') as file:
                        attack_loaded_data = pickle.load(file)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' + d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'random-attack'
                    self.data.append(harmful_data[self.cols])

            elif dataname == 'pseudo-attack':
                if type == 'train' or type =='valid':
                    import pickle
                    print(f" data loading from {self.data_folder_path}/{dataname}/train_All.pkl")
                    with open(f'{self.data_folder_path}/{dataname}/train_All.pkl','rb') as f:
                        attack_loaded_data = pickle.load(f)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' +d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'pseudo-attack'
                    from sklearn.model_selection import train_test_split
                    harmful_train, harmful_valid = train_test_split(harmful_data, test_size=0.25, random_state=0)
                    if type =='train':
                        self.data.append(harmful_train[self.cols])
                    elif type =='valid':
                        self.data.append(harmful_valid[self.cols])

                elif type =='test':
                    import pickle
                    print(f" data loading from {self.data_folder_path}/{dataname}/test_All.pkl")
                    with open(f'{self.data_folder_path}/{dataname}/test_All.pkl', 'rb') as file:
                        attack_loaded_data = pickle.load(file)
                    harmful_data = pd.DataFrame()
                    harmful_data['transcript'] = ['\n\nHuman:' + d[0] + ' ' + d[2] for d in attack_loaded_data]
                    harmful_data['rating'] = 1
                    harmful_data['datatype'] = 'pseudo-attack'
                    self.data.append(harmful_data[self.cols])

            else:
                raise TypeError(f"Data {dataname} does not exist")
        total_data = pd.concat(self.data , ignore_index=True)
        total_data['rating'] = total_data['rating'].map(lambda x : 1 if x >0 else 0) # assert binary
        return total_data



