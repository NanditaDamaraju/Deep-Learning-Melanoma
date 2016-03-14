#PBS -q class
#PBS -l nodes=jinx4
#PBS -l walltime=00:05:00
#PBS -N MyJobName
export LD_LIBRARY_PATH=$HOME/python/lib/:$LD_LIBRARY_PATH
export PATH=$HOME/python/Python-2.7.2/:$HOME/cmake/bin/:$PATH
export PYTHONPATH=$PYTHONPATH:/nethome/magarwal37/lib/python2.7/site-packages/
export PYTHONPATH=$PYTHONPATH:/nethome/magarwal37/install/opencv/lib/python2.6/site-packages/

cd $PBS_O_WORKDIR
python /nethome/magarwal37/Github/DL8803/test/vgg-16_keras.py
