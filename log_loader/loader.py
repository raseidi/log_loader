import torch , itertools
from pandas import DataFrame
from pm4py.objects.log.obj import EventLog

class LogForML():
    '''
    Converts a log into a Dataset suitable for ML tasks.
    '''
    def __init__(self, log, prefix_len=5):
        self.prefix_len = prefix_len
        self.log_type = self.check_log(log)
        self._preprocess(log)

    def _preprocess(self, log):
        if self.log_type == 'pandas':
            raise 'Not supported yet. To be implemented.'
        elif self.log_type == 'pm4py':
            self._log, self._word_to_ix, self._ix_to_word = self.act_to_vec(log)
        
        self._to_xy()

    @staticmethod
    def check_log(log):
        assert isinstance(log, DataFrame) or isinstance(log, EventLog), 'Invalid log input. Please provide pandas.DataFrame or pm4py.objects.log.obj.EventLog objects.'
        return 'pandas' if isinstance(log, DataFrame) else 'pm4py'

    @staticmethod
    def act_to_vec(log):
        word_to_ix = {}
        vocab_size = 0
        vectorized_activities = []
        for trace in log:
            trace_ids = []
            for event in trace:
                if event['concept:name'] not in word_to_ix:
                    word_to_ix[event['concept:name']] = vocab_size
                    vocab_size += 1
                trace_ids.append(word_to_ix[event['concept:name']])
            if '[EOC]' not in word_to_ix:
                word_to_ix['[EOC]'] = vocab_size
                vocab_size += 1
            trace_ids.append(word_to_ix['[EOC]'])
            vectorized_activities.append(trace_ids)

        vectorized_activities = list(itertools.chain.from_iterable(vectorized_activities))
        ix_to_word = {v: k for k, v in word_to_ix.items()}
        return vectorized_activities, word_to_ix, ix_to_word

    def _to_xy(self):
        self._X = []
        self._y = []
        for i in range(0, len(self._log), self.prefix_len):
            if len(self._log[i+1:i+1+self.prefix_len]) != self.prefix_len:
                continue
            self._X.append(self._log[i:i+self.prefix_len])
            self._y.append(self._log[i+self.prefix_len]) #+1:i+1+self.args.num_steps])
        
        self._X = torch.tensor(self._X, dtype=torch.long)
        self._y = torch.tensor(self._y, dtype=torch.long)
        del self._log

    def get_unique_actvities(self):
        if not hasattr(self, 'unique'):
            self.unique = sorted(self._word_to_ix.keys())
        return self.unique

    def get_vocab_size(self):
        return len(self.get_unique_actvities())

    def get_xy(self):
        return self._X, self._y

    def __len__(self):
        return len(self._X)

    def __getitem__(self, index):
        return (
            self._X[index], 
            self._y[index]
        )