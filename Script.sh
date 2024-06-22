#!/bin/bash

# Function to launch the script with given arguments
launch_script() {
  local python_path="$1"
  local script_path="$2"
  local dataset="$3"
  local classes="$4"
  local data_size="$5"
  local data_path="$6"
  local epochs="$7"
  local device="$8"
  local train_test="$9"
  local batch_size="${10}"
  local top_k="${11}"
  local results_path="${12}"
  local wandb_key="${13}"
  local model="${14}"
  local loss="${15}"
  local learn_rate="${16}"
  local hidden="${17}"
  local dropout="${18}"
  local regularization="${19}"
  local ntoken="${20}"
  local token_overlap="${21}"
  local learn_features="${22}"
  local model_id="${23}"
  local classifier="${24}"
  local nhead="${25}"
  local encoder_layers="${26}"
  local feedforward="${27}"
  local learn_tokens="${28}"
  local patience="${29}"

  # Construct the id_suffix
  local id_suffix="${model_id}_${model}_${hidden}"
  if [ "$learn_tokens" = "true" ]; then
    id_suffix="${id_suffix}_tkn"
  fi

  # Construct the command
  local command="$python_path $script_path -d $dataset -c $classes -dt $data_size -dp $data_path -e $epochs -dv $device -pt $train_test -bc $batch_size -tk $top_k -rp $results_path -wk $wandb_key -m $model -l $loss -lr $learn_rate -hn $hidden -dr $dropout -re $regularization -nt $ntoken -to $token_overlap -id $id_suffix -cl $classifier -nh $nhead -en $encoder_layers -ff $feedforward -pat $patience"

  if [ "$learn_features" = "true" ]; then
    command="$command -learn_features"
  fi

  if [ "$learn_tokens" = "true" ]; then
    command="$command -learn_tokens"
  fi

  # Launch the process
  echo "Launching command: $command"
  $command
}

# Default values
PYTHON_PATH="/home/ale_piazza/.conda/envs/ale_piazza/bin/python"
SCRIPT_PATH="/home/ale_piazza/MasterThesis/Train_token.py"
DATASET=("NTU")
CLASSES=("60")
DATA_SIZE=("Full")
DATA_PATH="/data03/Users/Alessandro/Data/"
EPOCHS=("50")
DEVICE=("cuda:1")
TRAIN_TEST=("0.7")
BATCH_SIZE=("16")
TOP_K=("5")
RESULTS_PATH="/home/ale_piazza/MasterThesis/Results"
WANDB_KEY="5296fdee5335cb67e4fc2e0feb6985ed78cca00a"
MODEL=("GCN")
LOSS=("CE")
LEARN_RATE=("0.001")
HIDDEN=("32" "64" "128")
DROPOUT=("0.1")
REGULARIZATION=("0.000000001")
NTOKEN=("10")
TOKEN_OVERLAP=("4")
LEARN_FEATURES=("true")  # Use "true" or "false"
CLASSIFIER=("ClassifierWin")
NHEAD=("8")
ENCODER_LAYERS=("6")
FEEDFORWARD=("2048")
LEARN_TOKENS=("true" "false")  # Use "true" or "false"
PATIENCE=("3")

# Model identifier prefix (static part)
MODEL_ID_PREFIX="_XSub"

# Loop over all combinations of the parameters
for dataset in "${DATASET[@]}"; do
  for classes in "${CLASSES[@]}"; do
    for data_size in "${DATA_SIZE[@]}"; do
      for epochs in "${EPOCHS[@]}"; do
        for device in "${DEVICE[@]}"; do
          for train_test in "${TRAIN_TEST[@]}"; do
            for batch_size in "${BATCH_SIZE[@]}"; do
              for top_k in "${TOP_K[@]}"; do
                for model in "${MODEL[@]}"; do
                  for loss in "${LOSS[@]}"; do
                    for learn_rate in "${LEARN_RATE[@]}"; do
                      for hidden in "${HIDDEN[@]}"; do
                        for dropout in "${DROPOUT[@]}"; do
                          for regularization in "${REGULARIZATION[@]}"; do
                            for ntoken in "${NTOKEN[@]}"; do
                              for token_overlap in "${TOKEN_OVERLAP[@]}"; do
                                for learn_features in "${LEARN_FEATURES[@]}"; do
                                  for classifier in "${CLASSIFIER[@]}"; do
                                    for nhead in "${NHEAD[@]}"; do
                                      for encoder_layers in "${ENCODER_LAYERS[@]}"; do
                                        for feedforward in "${FEEDFORWARD[@]}"; do
                                          for learn_tokens in "${LEARN_TOKENS[@]}"; do
                                            for patience in "${PATIENCE[@]}"; do
                                              # Construct model_id
                                              MODEL_ID="${MODEL_ID_PREFIX}"
                                              launch_script "$PYTHON_PATH" "$SCRIPT_PATH" "$dataset" "$classes" "$data_size" "$DATA_PATH" "$epochs" "$device" "$train_test" "$batch_size" "$top_k" "$RESULTS_PATH" "$WANDB_KEY" "$model" "$loss" "$learn_rate" "$hidden" "$dropout" "$regularization" "$ntoken" "$token_overlap" "$learn_features" "$MODEL_ID" "$classifier" "$nhead" "$encoder_layers" "$feedforward" "$learn_tokens" "$patience"
                                            done
                                          done
                                        done
                                      done
                                    done
                                  done
                                done
                              done
                            done
                          done
                        done
                      done
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
