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
from model.deepfm import DeepFM
from data_loader.dataloader import Feature
from common.utils import make_feature
optuna.logging.set_verbosity(optuna.logging.ERROR)
logging.getLogger().setLevel(logging.DEBUG)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("git").setLevel(logging.WARNING)


class Deepfm_Tuning:

    def __init__(self):
        self.device = "cpu"

    def train(self, params, trial):
        """
        train function
        """
        auroc = BinaryAUROC(thresholds=5).to(self.device)

        batch_size = params["batch_size"]
        number_epochs = params["number_epochs"]
        train_ratio = params["train_ratio"]
        learning_rate = params["learning_rate"]
        embedding_size = params["embedding_size"]
        dropout = params["dropout"]
        concat_df, feature_columns = make_feature()
        field_index = list(range(0, len(feature_columns)))
        field_dict = dict(zip(field_index, feature_columns))

        # load dataset
        dataset = Feature(dataset=concat_df)

        # 총 데이터 수
        dataset_size = len(dataset)
        params["dataset_size"] = dataset_size
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
            batch_size=batch_size
        )

        # 검증 데이터 로더는 워커 2
        validation_loader = DataLoader(
            dataset=validation_dataset,
            shuffle=False,
            batch_size=validation_size
        )

        test_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=test_size
        )

        # set model
        model = DeepFM(
            embedding_size=embedding_size,
            number_feature=len(field_index),
            number_field=len(field_dict),
            field_index=field_index,
            dropout=float(dropout)
        ).to(self.device)

        optimization_function = Adam(model.parameters(), learning_rate)

        # scheduler
        scheduler = StepLR(optimizer=optimization_function,
                           step_size=10,
                           gamma=0.1)

        # mlflow
        mlflow.set_tracking_uri("localhost")

        if not mlflow.get_experiment_by_name("DeepFM_MovieLens"):
            experiment_id = mlflow.create_experiment(name="DeepFM_MovieLens")
            experiment = mlflow.get_experiment(experiment_id)
        else:
            experiment_id = mlflow.get_experiment_by_name("DeepFM_MovieLens").experiment_id
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

                    question, response = data

                    model.train()

                    question = question.to(self.device)
                    response = response.to(self.device)

                    predict = model(question, self.device)
                    true_score = torch.squeeze(response).float().to(self.device)

                    optimization_function.zero_grad()

                    # NAN 처리
                    if int(torch.sum(torch.isnan(predict)).item()) >= 1:
                        predict = torch.nan_to_num(predict)
                    if int(torch.sum(torch.isnan(true_score)).item()) >= 1:
                        true_score = torch.nan_to_num(true_score)

                    loss = binary_cross_entropy(predict.to(torch.float64), true_score.to(torch.float64))
                    loss.backward()

                    optimization_function.step()

                    if len(set(true_score.detach().cpu().numpy())) == 2:
                        # save accuracy
                        accuracy = auroc(predict, true_score)
                        train_accuracies.append(float(accuracy.detach().cpu().numpy()))

                        # save loss
                        train_losses.append(float(loss.detach().cpu().numpy()))

                # start validation
                with torch.no_grad():
                    for data in validation_loader:

                        question, response = data

                        model.eval()

                        question = question.to(self.device)
                        response = response.to(self.device)

                        predict = model(question, self.device)
                        true_score = torch.squeeze(response).float().to(self.device)

                        # NAN 처리
                        if int(torch.sum(torch.isnan(predict)).item()) >= 1:
                            predict = torch.nan_to_num(predict)
                        if int(torch.sum(torch.isnan(true_score)).item()) >= 1:
                            true_score = torch.nan_to_num(true_score)

                        loss = binary_cross_entropy(predict.to(torch.float64), true_score.to(torch.float64))

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

                # 일반화 성능 검증
                with torch.no_grad():
                    for data in test_loader:

                        question, response = data

                        model.eval()

                        question = question.to(self.device)
                        response = response.to(self.device)

                        predict = model(question, self.device)
                        true_score = torch.squeeze(response).float().to(self.device)

                        # NAN 처리
                        if int(torch.sum(torch.isnan(predict)).item()) >= 1:
                            predict = torch.nan_to_num(predict)
                        if int(torch.sum(torch.isnan(true_score)).item()) >= 1:
                            true_score = torch.nan_to_num(true_score)

                        loss = binary_cross_entropy(predict.to(torch.float64), true_score.to(torch.float64))

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
    params = {
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 1.0),
        "embedding_size": trial.suggest_int("embedding_size", 2, 100),
        "batch_size": trial.suggest_int("batch_size", 1, 1024),
        "train_ratio": 0.8,
        "number_epochs": 50
    }

    result = Deepfm_Tuning().train(params=params, trial=trial)
    return result
