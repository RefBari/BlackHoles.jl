# Running Black Holes on GPUs

GPU: Graphics Processing Unit
CPU: Central Processing Unit

In a login node in Oscar: To run your scripts, you can request a batch job via a Scheduler like slurm. Slurm is a set of instructions (i.e., a script). It determines when you get the job amongst several requests. Then your job starts running on the compute nodes. 

How do you run your program on Oscar? 
> - Interactive Jobs: Program runs in dedicated terminal window
> - Batch jobs: Run program in background

Slurm needs to know the following before it can schedule your job, whether it's interactive or batch: 
> - How many nodes and cores do you need?
> - How much memory do you need?
> - Do you require a GPU?
> - How much time does your job need?

To run interactive jobs: 
`interact -q gpu -g 1 -n 1 -m 32g -t 1:00:00`
Where `-g` is the number of GPUs, `-n` is the number of cores, `-m` is the memory expected to be used by the job, and `-t` is the time. If you leave all the parameters blank, you'll get a default of 4 GB of memory and 30 minutes of time on the GPU. 

To submit your batch file, do `sbatch filename.sh`. To check on your jobs, use `myjobinfo`. Or, alternatively, `myjobinfo -j <JobID>`
