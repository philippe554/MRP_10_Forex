#!/usr/bin/env zsh

### Vars
JOB_DIR = $HOME/MRP_10_Forex/Jobs/
OUTPUT_DIR = Output/%J/

### Ensure that you use DKE resources
#BSUB -P um_dke

### Job name
#BSUB -J Testing_MRP10

#BSUB -B
#BSUB -N
#BSUB -u cjnj.kerkhofs@student.maastrichtuniversity.nl

### Create output dir
if [ ! -d "$OUTPUT_DIR" ]
then mkdir "$OUTPUT_DIR"
fi

### File / path where STDOUT will be written, the %J is the job id
#BSUB -o %OUTPUT_DIR/stdout

### Request the time you need for execution in minutes
### The format for the parameter is: [hour:]minute,
### that means for 80 minutes you could also use this: 1:20
#BSUB -W 12:00

### Request memory you need for your job in MB
#BSUB -M 4000

### Change to the work directory
cd $HOME/MRP_10_Forex/App/Library/lstm

### load modules and execute
module switch intel gcc
module load python/3.6.0
export PATH=$HOME/.local/bin:$PATH

# start non-interactive batch job
echo "Starting lstm training"
python3.6 lstm.py -p -n -m "%JOB_DIR%OUTPUT_DIR" -i "%HOME/Data/price_hist_ta.db" -f simple -c "%HOME/Temp"