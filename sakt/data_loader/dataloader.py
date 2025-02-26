from torch.utils.data import Dataset
import polars as pl

# 패키지 초기화 함수 불러오기
from tuning.sakt.common.utils import make_sequences, match_sequence_length


class Soft_Toc_Toc(Dataset):

    def __init__(self, sequence_length) -> None:
        dataset_pl = pl.read_csv("dataset.csv")
        dictionary_df = dataset_pl.select("assessmentItemID")
        dictionary_df = dictionary_df.unique(maintain_order=True).to_pandas()
        question_to_index = dict(zip(dictionary_df['assessmentItemID'], dictionary_df.index))

        dataset_pl = dataset_pl.sample(n=10000, seed=0).to_pandas()

        # 문항 번호 변환
        dataset_pl['assessmentItemID'] = dataset_pl['assessmentItemID'].apply(lambda x: question_to_index[x])

        # 시퀀스 변환
        question_sequences, response_sequences = make_sequences(dataframe=dataset_pl)

        # 패딩
        self.question_sequences, self.response_sequences = match_sequence_length(
            question_sequences=question_sequences,
            response_sequences=response_sequences,
            sequence_length=sequence_length
        )
        self.number_question = len(dictionary_df)
        self.length = len(question_sequences)

    def __getitem__(self, index):
        return self.question_sequences[index], self.response_sequences[index]

    def __len__(self):
        return self.length
