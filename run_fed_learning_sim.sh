#!/bin/bash

# Definisci il percorso assoluto alla directory dei dati
DATA_DIR="../../datasets/OASIS_MRI_splitted"
SERVER_BASE_DIR="../server_base"
CLIENTS_BASE_DIR="../clients_base"
# Imposta il PYTHONPATH
export PYTHONPATH=$PYTHONPATH:fedLearning/mri

# Avvia il server prima, passando il percorso dei dati per il test e il percorso di salvataggio
gnome-terminal -- bash -c "python3 blenderLauncher.py server --data_type mri --data_path '$DATA_DIR/subset_0' --base_dir $SERVER_BASE_DIR --classes Non_Demented Very_mild_Dementia; exec bash"

# Avvia i client successivamente, con ID da 1 a 2, passando il percorso dei dati per ciascun client
for i in {1..2}; do
    gnome-terminal -- bash -c "python3 blenderLauncher.py client --data_type mri --data_path '$DATA_DIR/subset_$i' --base_dir $CLIENTS_BASE_DIR/client_base_$i --client_id $i --classes Non_Demented Very_mild_Dementia; exec bash"
done
