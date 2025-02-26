import mlflow
import statistics
import optuna
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.functional import binary_cross_entropy
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.classification import BinaryAUROC
import logging
from data_loader.dataloader import Soft_Toc_Toc
from common.utils import collate_fn, find_multiples
from model.sakt import SAKT
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)


class Sakt_Tuning:

    def __init__(self):
        self.device = "cpu"

    def tuning(self, params, trial):
        """
        tuning function
        """
        auroc = BinaryAUROC(thresholds=5).to(self.device)

        # load dataset
        dataset = Soft_Toc_Toc(sequence_length=params['sequence_length'])
        number_questions = dataset.number_question
        params['number_questions'] = number_questions

        # set config
        batch_size = params['batch_size']
        number_epochs = params['number_epochs']
        train_ratio = params['train_ratio']
        learning_rate = params['learning_rate']
        dropout = params['dropout']
        n = params['sequence_length']
        d = params['sequence_length']
        number_attention_heads = params['number_attention_heads']

        # 총 데이터 수
        dataset_size = len(dataset)

        # 훈련 데이터 수
        train_size = int(dataset_size * float(train_ratio))

        # 검증 데이터 수
        validation_size = int(dataset_size * 0.1)

        # 데스트 데이터 수 (일반화 성능 측정)
        test_size = dataset_size - train_size - validation_size

        # random_split 활용
        train_dataset, validation_dataset, test_dataset = random_split(
            dataset, [train_size, validation_size, test_size])

        # set dataloader
        train_loader = DataLoader(
            dataset=train_dataset,
            shuffle=True,
            batch_size=batch_size,
            collate_fn=collate_fn
        )

        # 검증 데이터 로더는 워커 2
        validation_loader = DataLoader(
            dataset=validation_dataset,
            shuffle=False,
            batch_size=validation_size,
            collate_fn=collate_fn
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=test_size,
            collate_fn=collate_fn
        )

        # set model
        model = SAKT(
            number_questions=int(number_questions),
            n=int(n),
            d=int(d),
            number_attention_heads=int(number_attention_heads),
            dropout=float(dropout)
        ).to(self.device)

        # set optimization_function
        optimization_function = Adam(model.parameters(), learning_rate)

        # scheduler
        scheduler = StepLR(optimizer=optimization_function,
                           step_size=10,
                           gamma=0.1)

        # mlflow
        mlflow.set_tracking_uri("localhost")

        if not mlflow.get_experiment_by_name("SAKT_all"):
            experiment_id = mlflow.create_experiment(name="SAKT_all")
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment_id = mlflow.get_experiment_by_name("SAKT_all").experiment_id
            experiment = mlflow.get_experiment(experiment_id)

        test_accuracies = []
        with mlflow.start_run(experiment_id=experiment.experiment_id):
            mlflow.log_params(params)
            for epoch in range(1, number_epochs):

                train_losses = []
                train_accuracies = []
                vaild_losses = []
                vaild_accuracies = []
                test_losses = []

                # start train
                torch.cuda.empty_cache()
                for data in train_loader:

                    question, response, question_shift, response_shift, masked = data

                    model.train()

                    question = question.to(self.device)
                    response = response.to(self.device)
                    question_shift = question_shift.to(self.device)
                    masked = masked.to(self.device)
                    response_shift = response_shift.to(self.device)

                    predict, _ = model(question, response, question_shift, self.device)

                    predict = torch.masked_select(predict, masked).to(self.device)
                    true_score = torch.masked_select(response_shift, masked).float().to(self.device)

                    optimization_function.zero_grad()

                    loss = binary_cross_entropy(predict.to(torch.float64), true_score.to(torch.float64))
                    loss.backward()

                    optimization_function.step()

                    if len(set(true_score.detach().cpu().numpy())) == 2:
                        # save accuracy
                        accuracy = auroc(predict, true_score)
                        train_accuracies.append(float(accuracy.detach().cpu().numpy()))

                        # save loss
                        train_losses.append(float(loss.detach().cpu().numpy()))

                torch.cuda.empty_cache()
                # start validation
                with torch.no_grad():
                    for data in validation_loader:

                        question, response, question_shift, response_shift, masked = data

                        model.eval()

                        question = question.to(self.device)
                        response = response.to(self.device)
                        question_shift = question_shift.to(self.device)
                        masked = masked.to(self.device)
                        response_shift = response_shift.to(self.device)

                        predict, _ = model(question, response, question_shift, self.device)
                        predict = torch.masked_select(predict, masked).to(self.device)
                        true_score = torch.masked_select(response_shift, masked).float().to(self.device)

                        if len(set(true_score.detach().cpu().numpy())) == 2:
                            # save accuracy
                            accuracy = auroc(predict, true_score)
                            vaild_accuracies.append(float(accuracy.detach().cpu().numpy()))

                            # save loss
                            vaild_losses.append(float(loss.detach().cpu().numpy()))

                trial.report(statistics.mean(vaild_accuracies), epoch)

                if trial.should_prune():
                    logging.debug("prune")
                    mlflow.set_tag("State", "Early_Stop")
                    raise optuna.TrialPruned()

                with torch.no_grad():
                    for data in test_loader:

                        question, response, question_shift, response_shift, masked = data

                        model.eval()

                        question = question.to(self.device)
                        response = response.to(self.device)
                        question_shift = question_shift.to(self.device)
                        masked = masked.to(self.device)
                        response_shift = response_shift.to(self.device)

                        predict, _ = model(question, response, question_shift, self.device)
                        predict = torch.masked_select(predict, masked).to(self.device)
                        true_score = torch.masked_select(response_shift, masked).float().to(self.device)

                        if len(set(true_score.detach().cpu().numpy())) == 2:
                            # save accuracy
                            accuracy = auroc(predict, true_score)
                            test_accuracies.append(float(accuracy.detach().cpu().numpy()))

                            # save loss
                            test_losses.append(float(loss.detach().cpu().numpy()))
                # mlflow logging
                mlflow.log_metrics({
                    "train_loss": statistics.mean(train_losses),
                    "train_accuracy": statistics.mean(train_accuracies),
                    "vaild_loss": statistics.mean(vaild_losses),
                    "vaild_accuracy": statistics.mean(vaild_accuracies),
                    "test_loss": statistics.mean(test_losses),
                    "test_accuracy": statistics.mean(test_accuracies)
                })
                mlflow.set_tag("State", "Done")
                logging.debug(
                    "Validation Epoch: {}, AUC: {}, Loss Mean: {}"
                    .format(epoch, statistics.mean(vaild_accuracies), statistics.mean(vaild_losses))
                )
                logging.debug(
                    "Test Epoch: {}, AUC: {}, Loss Mean: {}"
                        .format(epoch, statistics.mean(test_accuracies), statistics.mean(test_losses))
                )
            scheduler.step()
            return statistics.mean(test_accuracies)


# 목적함수 샘플링 알고리즘 정의
def objective(trial):

    # 1부터 100까지의 21의 배수 찾기
    resusequence_length_list = find_multiples(9, 13)

    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'dropout': trial.suggest_float('dropout', 0.0, 1.0),
        'batch_size': trial.suggest_int('batch_size', 1, 1024),
        'number_attention_heads': 21,
        'train_ratio': 0.8,
        'number_epochs': 50,
        'sequence_length': trial.suggest_categorical("sequence_length", resusequence_length_list)
    }

    result = Sakt_Tuning().tuning(params=params, trial=trial)
    return result
