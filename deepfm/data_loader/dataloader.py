import numpy as np
from torch.utils.data import Dataset


class Feature(Dataset):

    def __init__(self, dataset) -> None:
        super().__init__()

        dataset_df = dataset.sample(n=100000)

        # 'rating' 열을 제외한 나머지 열 이름 목록 생성
        selected_feature = [column for column in dataset_df.columns if column != "rating"]

        self.questions = dataset_df[selected_feature]
        # self.questions = self.questions.to_pandas()
        
        self.responses = dataset_df["rating"]
        # self.responses = self.responses.to_pandas()
    
        self.questions.index = range(0, len(self.questions))
        self.responses.index = range(0, len(self.responses))
        
        self.length = len(dataset_df)
        
    def __getitem__(self, index):

        question = np.array(self.questions.loc[index]).astype(np.float32)
        response = np.array(self.responses.loc[index]).astype(np.float32)
        
        return question, response

    def __len__(self):
        return self.length
    