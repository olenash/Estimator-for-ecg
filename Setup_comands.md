### Commands to run at Google Cloud VM
```
PROJECT_ID=datasciencegroup-216409
BUCKET_NAME=ecg-denoise-demo-mlengine
echo $BUCKET_NAME
REGION=europe-west1
```

```
TRAIN_CLEAN_DATA=temp_trY.pklz
TRAIN_NOISY_DATA=temp_trX.pklz
EVAL_CLEAN_DATA=temp_teY.pklz
EVAL_NOISY_DATA=temp_teX.pklz
```

```
DATE=`date '+%Y%m%d_%H%M%S'`
export JOB_NAME=ecg_$DATE
OUTPUT_PATH=gs://$BUCKET_NAME/$JOB_NAME
echo $OUTPUT_PATH
```
Train localy
```
gcloud ml-engine local train \
--module-name trainer.task \
--package-path trainer/ \
-- \
--train-clean-files $TRAIN_CLEAN_DATA \
--train-noisy-files $TRAIN_NOISY_DATA \
--eval-clean-files $EVAL_CLEAN_DATA \
--eval-noisy-files $EVAL_NOISY_DATA \
--train-steps 100 \
--job-dir $OUTPUT_PATH \
```

### Deploy job

```
TRAIN_CLEAN_DATA=gs://$BUCKET_NAME/data/temp_trY.pklz
TRAIN_NOISY_DATA=gs://$BUCKET_NAME/data/temp_trX.pklz
EVAL_CLEAN_DATA=gs://$BUCKET_NAME/data/temp_teY.pklz
EVAL_NOISY_DATA=gs://$BUCKET_NAME/data/temp_teX.pklz
```

```
gcloud ml-engine jobs submit training $JOB_NAME \
--job-dir $OUTPUT_PATH \
--runtime-version 1.10 \
--module-name trainer.task \
--package-path trainer/ \
--region $REGION \
-- \
--train-clean-files $TRAIN_CLEAN_DATA \
--train-noisy-files $TRAIN_NOISY_DATA \
--eval-clean-files $EVAL_CLEAN_DATA \
--eval-noisy-files $EVAL_NOISY_DATA \
--train-steps 500
```
```
gcloud ml-engine jobs stream-logs $JOB_NAME
```
```
gsutil ls -r $OUTPUT_PATH
```
```
tensorboard --logdir=$OUTPUT_PATH --port=8080
```

```
gcloud compute ssh [instance name] -- -L 5000:localhost:5000
```