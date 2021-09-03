# define base dir
export SQUAD_DIR=.

# train
python run_squad.py \
  --bert_model bert-base-uncased \
  --squad_model bert_deep \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/data/train-v2.0.json \
  --predict_file $SQUAD_DIR/data/dev-v2.0.json \
  --train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir $SQUAD_DIR/output

# wdevaluate
python evaluate-v2.0.py $SQUAD_DIR/data/dev-v2.0.json \
    $SQUAD_DIR/output/predictions.json \
    --out-file $SQUAD_DIR/output/dev_eval.json \
    --na-prob-file $SQUAD_DIR/output/null_odds.json \
    --out-image-dir $SQUAD_DIR/output/charts

# generate dev submission file
python submission.py --pred_file $SQUAD_DIR/output/predictions.json \
    --out-file $SQUAD_DIR/output/dev_submission.csv

# rename dev predictions
mv $SQUAD_DIR/output/predictions.json $SQUAD_DIR/output/dev_predictions.json
mv $SQUAD_DIR/output/nbest_predictions.json $SQUAD_DIR/output/dev_nbest_predictions.json   
mv $SQUAD_DIR/output/null_odds.json $SQUAD_DIR/output/dev_null_odds.json

# test
python run_squad.py \
  --bert_model bert-base-uncased \
  --squad_model bert_deep \
  --do_predict \
  --do_lower_case \
  --predict_file $SQUAD_DIR/data/test-v2.0.json \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir $SQUAD_DIR/output/

# generate test submission file
python submission.py --pred_file $SQUAD_DIR/output/predictions.json \
    --out-file $SQUAD_DIR/output/test_submission.csv

# rename dev predictions
mv $SQUAD_DIR/output/predictions.json $SQUAD_DIR/output/test_predictions.json   
mv $SQUAD_DIR/output/nbest_predictions.json $SQUAD_DIR/output/test_nbest_predictions.json   
mv $SQUAD_DIR/output/null_odds.json $SQUAD_DIR/output/test_null_odds.json


# save config under output
cp $SQUAD_DIR/run.sh $SQUAD_DIR/output/


