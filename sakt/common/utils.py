import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


def make_sequences(dataframe):
    # 학생별 그룹바이
    groupby_df = dataframe.groupby('member_id')

    result = groupby_df['question_id'].apply(list)
    question_sequences = result.tolist()

    result = groupby_df['answer'].apply(list)
    response_sequences = result.tolist()

    return question_sequences, response_sequences


def match_sequence_length(question_sequences: list,
                          response_sequences: list,
                          sequence_length: int,
                          padding_value=-1
                          ):
    """Function that matches the length of question_sequences and response_sequences to the length of sequence_length.

    Args:
        question_sequences: A list of question solutions for each student.
        response_sequences: A list of questions and answers for each student.
        sequence_length: Length of sequence.
        padding_value: Value of padding.

    Returns:
        length-matched parameters.

    Note:
        Return detail.

        - proc_question_sequences : length-matched question_sequences.
        - proc_response_sequences : length-matched response_sequences.

    Examples:
        >>> match_sequence_length(question_sequences=[[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18]...],
        >>>                       response_sequence=[[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0]...],
        >>>                       sequence_length=50,
        >>>                       padding_value=-1)
        ([[67 18 67 18 67 18 18 67 67 18 18 35 18 32 18 -1 -1 -1 ... -1 -1 -1]...],
        [[0 0 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 0 0 -1 -1 -1 ... -1 -1 -1]...])
    """
    proc_question_sequences = []
    proc_response_sequences = []
    proc_user_list = []

    for question_sequence, response_sequence in zip(question_sequences, response_sequences):

        i = 0

        while i + sequence_length + 1 < len(question_sequence):
            proc_question_sequences.append(question_sequence[i:i + sequence_length + 1])
            proc_response_sequences.append(response_sequence[i:i + sequence_length + 1])
            i += sequence_length + 1

        proc_question_sequences.append(
            np.concatenate(
                [
                    question_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

        proc_response_sequences.append(
            np.concatenate(
                [
                    response_sequence[i:],
                    np.array([padding_value] * (i + sequence_length + 1 - len(question_sequence)))
                ]
            )
        )

    return proc_question_sequences, proc_response_sequences


def collate_fn(
        batch,
        padding_value=-1
):
    """The collate function for torch.utils.data.DataLoader

    Args:
        batch: data batch.
        padding_value: Value of padding.

    Returns:
        Dataloader elements for model training.

    Note:
        Return detail.

        - question_sequences: the question(KC) sequences.
            - question_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_sequences: the response sequences.
            - response_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - question_shift_sequences: the question(KC) sequences which were shifted one step to the right.
            - question_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - response_shift_sequences: the response sequences which were shifted one step to the right.
            - response_shift_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].

        - mask_sequences: the mask sequences indicating where the padded entry.
            - mask_sequences_shape: [batch_size, maximum_sequence_length_in_the_batch].
    """
    question_sequences = []
    response_sequences = []
    question_shift_sequences = []
    response_shift_sequences = []

    for q_seq, r_seq in batch:
        question_sequences.append(torch.tensor(q_seq[:-1], dtype=torch.float64, device='cpu'))
        response_sequences.append(torch.tensor(r_seq[:-1], dtype=torch.float64, device='cpu'))
        question_shift_sequences.append(torch.tensor(q_seq[1:], dtype=torch.float64, device='cpu'))
        response_shift_sequences.append(torch.tensor(r_seq[1:], dtype=torch.float64, device='cpu'))

    question_sequences = pad_sequence(
        question_sequences, batch_first=True, padding_value=padding_value
    )
    response_sequences = pad_sequence(
        response_sequences, batch_first=True, padding_value=padding_value
    )
    question_shift_sequences = pad_sequence(
        question_shift_sequences, batch_first=True, padding_value=padding_value
    )
    response_shift_sequences = pad_sequence(
        response_shift_sequences, batch_first=True, padding_value=padding_value
    )

    mask_sequences = (question_sequences != padding_value) * (question_shift_sequences != padding_value)

    question_sequences, response_sequences, question_shift_sequences, response_shift_sequences = \
        question_sequences * mask_sequences, response_sequences * mask_sequences, question_shift_sequences * mask_sequences, \
        response_shift_sequences * mask_sequences

    return question_sequences, response_sequences, question_shift_sequences, response_shift_sequences, mask_sequences


def find_multiples(start, end):
    multiples = []
    for num in range(start, end + 1):
        multiples.append(num * 21)
    return multiples



