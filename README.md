# Run MUMPS comparison on Python

## install MUMPS

    git clone https://github.com/scivision/mumps.git
    cd mumps 
    cmake -Bbuild -DBUILD_SINGLE=on -DBUILD_DOUBLE=on -DBUILD_COMPLEX=on -DBUILD_COMPLEX16=on -DMUMPS_parallel=no -DBUILD_SHARED_LIBS=on  -DMUMPS_UPSTREAM_VERSION=5.7.3 --install-prefix=XXXX/envs/YYY/
    cmake --build build -j20
    cmake --install build 

sudo can be used for the last command depending on prefix
prefix must be adapted especillay in the case of use of Python's environement (using `python -c "import sysconfig;print(sysconfig.get_path('data'))"`)

## install python packages

    pip install -r requirements.txt

## run script

    python compare.py