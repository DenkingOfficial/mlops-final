from airflow import DAG
from airflow.operators.bash import BashOperator
import datetime as dt

args = {
    "owner": "admin",
    "start_date": dt.datetime(2024, 1, 14, 20, 40),
    "retries": 1,
    "retry_delays": dt.timedelta(minutes=1),
    "depends_on_past": False,
}

with DAG(
    "Text-Training",
    description="Text binary classification",
    schedule_interval="*/1 * * * *",
    default_args=args,
    tags=["text", "classification"],
) as dag:
    data_download = BashOperator(
        task_id="data_download",
        bash_command="python3 /home/mlserv/mlops-final/scripts/data_download.py",
        dag=dag,
    )
    data_preprocess = BashOperator(
        task_id="data_preprocess",
        bash_command="python3 /home/mlserv/mlops-final/scripts/data_preprocess.py",
        dag=dag,
    )
    model_train = BashOperator(
        task_id="model_train",
        bash_command="python3 /home/mlserv/mlops-final/scripts/model_train.py",
        dag=dag,
    )
    model_test = BashOperator(
        task_id="model_test",
        bash_command="python3 /home/mlserv/mlops-final/scripts/model_test.py",
        dag=dag,
    )
    data_download >> data_preprocess >> model_train >> model_test