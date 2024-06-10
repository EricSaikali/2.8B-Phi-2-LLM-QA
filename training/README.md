**In the first terminal:**
- `$username="<username>"`
- `$repo_path="<path_to_repo>"`
- `ssh $username@izar.epfl.ch`

**In different terminal:**
- `scp -r $repo_path $username@izar:/scratch/izar/$username/`

**In the first terminal:**
- `cd /scratch/izar/$username/$repo_path`
- `cd training`
- `module load gcc python`
- `python -m venv /path/to/new/virtual/environment`
- `source /path/to/new/virtual/environment/bin/activate`
- `pip install -r requirements.txt`
- `pip install -U git+https://github.com/huggingface/trl`