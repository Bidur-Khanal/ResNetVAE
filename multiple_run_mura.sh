#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"mura"
epochs=$"300"


for batch in 512 
do 
    for lr in 0.001
    do 
        for version in 1 2
        do
            EXPT=VAE_mura_"$lr"_"$batch"_"$epochs"_"$version"
            STD=$dir/STD_VAE_mura_"$lr"_"$batch"_"$epochs"_"$version".out
            ERR=$dir/ERR_VAE_mura_"$lr"_"$batch"_"$epochs"_"$version".err
            export lr;
            export batch;
            export epochs;
            export version;
            export dataset;
            sbatch -J $EXPT -o $STD -t 00-18:00:00 -e $ERR $job_File
        done;
    done;
done;


