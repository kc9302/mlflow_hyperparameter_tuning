import torch


class Factorization_Machine_Layer(torch.nn.Module):
    def __init__(self, number_feature, number_field, embedding_size, field_index):
        super(Factorization_Machine_Layer, self).__init__()
        self.embedding_size = embedding_size  # k: 임베딩 벡터의 차원(크기)
        self.number_feature = number_feature  # f: 원래 feature 개수
        self.number_field = number_field  # m: grouped field 개수
        self.field_index = field_index  # 인코딩된 X의 칼럼들이 본래 어디 소속이었는지

        # Parameters of Factorization_Machine_Layer
        self.w = torch.nn.Parameter(torch.normal(size=[self.number_feature],
                                                 mean=0.0,
                                                 std=1.0))
        self.V = torch.nn.Parameter(torch.normal(size=[self.number_field, self.embedding_size],
                                                 mean=0.0,
                                                 std=0.01))

    def forward(self, inputs):
        inputs_batch = torch.reshape(inputs, (-1, self.number_feature, 1))
        # Parameter V를 field_index에 맞게 복사하여 number_feature에 맞게 늘림
        # Embedding lookup using field_index
        embedding = self.V[self.field_index]

        # Deep Component Input
        # Deep Component에서 쓸 Input
        # (batch_size, number_feature, embedding_size)
        preprocessed_inputs = torch.mul(inputs_batch, embedding)  # Element-wise multiplication

        # Linear terms
        linear_terms = torch.mean(torch.multiply(self.w, inputs), dim=1, keepdim=True)

        # Interaction terms
        interactions = 0.5 * torch.subtract(
            torch.square(torch.mean(preprocessed_inputs, dim=[1, 2])),
            torch.mean(torch.square(preprocessed_inputs), dim=[1, 2])
        )
        interactions = torch.reshape(interactions, [-1, 1])
        factorization_machine_layer_out_put = torch.concat([linear_terms, interactions], dim=1)

        return factorization_machine_layer_out_put, preprocessed_inputs
