#!/bin/bash

bsub -env "all,MOL=H2,NoNoise=1,P=0.0" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=0.0" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=0.0" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=0.0" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 


bsub -env "all,MOL=H2,NoNoise=1,P=0.01" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=0.01" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=0.01" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=0.01" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 

bsub -env "all,MOL=H2,NoNoise=1,P=0.1" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=0.1" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=0.1" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=0.1" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 

bsub -env "all,MOL=H2,NoNoise=1,P=0.3" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=0.3" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=0.3" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=0.3" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 

bsub -env "all,MOL=H2,NoNoise=1,P=0.7" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=0.7" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=0.7" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=0.7" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 

bsub -env "all,MOL=H2,NoNoise=1,P=1.0" -J "H2[1-5]" -R "rusage[mem=2GB:swp=2GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=H2,NoNoise=0,P=1.0" -J "H2-Noisy[1-5]" -R "rusage[mem=2GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=LiH,NoNoise=1,P=1.0" -J "LiH[1-5]" -R "rusage[mem=10GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 
bsub -env "all,MOL=BeH2,NoNoise=1,P=1.0" -J "BeH2[1-5]" -R "rusage[mem=20GB:swp=5GB]" -R "span[hosts=1]" < scripts/submit-job.sh 


bsub -env "all,MOL=DMABN,NoNoise=1,P=1.0" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "16:00" < scripts/submit-job.sh 
bsub -env "all,MOL=DMABN,NoNoise=1,P=0.7" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "16:00" < scripts/submit-job.sh 
bsub -env "all,MOL=DMABN,NoNoise=1,P=0.3" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "16:00" < scripts/submit-job.sh 
bsub -env "all,MOL=DMABN,NoNoise=1,P=0.1" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "48:00" < scripts/submit-job.sh 
bsub -env "all,MOL=DMABN,NoNoise=1,P=0.01" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "48:00" < scripts/submit-job.sh 
bsub -env "all,MOL=DMABN,NoNoise=1,P=0.0" -J "DMABN[1-5]" -R "rusage[mem=100GB:swp=5GB]" -R "span[hosts=1]" -W "48:00" < scripts/submit-job.sh 