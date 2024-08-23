import subprocess
from argparse import ArgumentParser

def run(args):

    filename = generate_cluster_script(args.command,
        args.queue,
        args.gpus,
        args.cpus,
        args.walltime,
        args.memory,
        args.name,
        args.email)
    subprocess.run(f"bsub < {filename}", shell=True)

def generate_cluster_script(
    python_command,
    job_queue,
    num_gpus,
    num_cpus,
    walltime,
    memory,
    experiment_name,
    email
):
    script_template = f"""#!/bin/sh
### General options
### -- specify queue --
#BSUB -q {job_queue}
### -- set the job Name --
#BSUB -J {experiment_name}
### -- ask for number of cores (default: 1) --
#BSUB -n {num_cpus}
### -- specify that the cores MUST BE on a single host --
#BSUB -R "span[hosts=1]"
### -- Select the resources: {num_gpus} gpu(s) in exclusive process mode --
#BSUB -gpu "num={num_gpus}:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues right now
#BSUB -W {walltime}
### -- request memory --
#BSUB -R "rusage[mem={memory}]"
### -- set the email address --
#BSUB -u {email}
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o {experiment_name}_%J.out
#BSUB -e {experiment_name}_%J.err
# -- end of LSF options --

# Load the modules, and activate the environment if needed
# module load python3/...

# Activate the environment
source ~/envs/huggingface/bin/activate

# Run the python command
torchrun --standalone --nproc_per_node=1 {python_command}
"""
    filename = f"{experiment_name}_cluster_script.sh"
    # Write the script to a file
    with open(filename, "w") as file:
        file.write(script_template)

    print(f"Cluster script {experiment_name}_cluster_script.sh generated successfully!")
    return filename

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--command", type=str, required=True,
                        help="Command to be run on the cluster node")
    parser.add_argument("--queue", type=str, default="p1", help="The queue on the cluster to be used")
    parser.add_argument("--gpus", type=int, default=1, help="Number of gpus needed")
    parser.add_argument("--cpus", type=int, default=8, help="Number of cpu cores needed")
    parser.add_argument("--walltime", type=str, default="72:00", help="Maximum wallclock time in hours:minutes")
    parser.add_argument("--memory", type=str, default="64GB", help="Required RAM in MB")
    parser.add_argument("--name", type=str, default="layershuffle", help="experiment name")
    parser.add_argument("--email", type=str, default="mafr@di.ku.dk", help="mail address")

    args = parser.parse_args()
    run(args)










