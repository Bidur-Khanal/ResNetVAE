#! bin bash -l

dir="sbatch_log"
job_File="sbatch_run.sh" 
dataset=$"covid19_xray"
epochs=$"300"


for batch in 512 
do 
    for lr in 0.001
    do 
        for version in 1 2
        do
            EXPT=VAE_covid19_xray_"$lr"_"$batch"_"$epochs"_"$version"
            STD=$dir/STD_VAE_covid19_xray_"$lr"_"$batch"_"$epochs"_"$version".out
            ERR=$dir/ERR_VAE_covid19_xray_"$lr"_"$batch"_"$epochs"_"$version".err
            export lr;
            export batch;
            export epochs;
            export version;
            export dataset;
            sbatch -J $EXPT -o $STD -t 00-12:00:00 -e $ERR $job_File
        done;
    done;
done;


